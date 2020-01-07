#define GLM_FORCE_AVX
#include "TopLevelBVH.h"

#define USE_TOP_MBVH 1
#define USE_MBVH 0
#define REFIT 0

void rfw::TopLevelBVH::constructBVH()
{
#if REFIT
	if (instanceCountChanged) // (Re)build
	{
#endif
		m_Nodes.clear();
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.grow(aabb);

		m_Nodes.resize(boundingBoxes.size() * 2);
		m_PrimIndices.resize(boundingBoxes.size());
		for (uint i = 0, s = static_cast<uint>(m_PrimIndices.size()); i < s; i++)
			m_PrimIndices[i] = i;

		m_PoolPtr.store(2);
		m_Nodes[0].bounds = rootBounds;
		m_Nodes[0].bounds.leftFirst = 0;
		m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());

		// Use less split bins to speed up construction
		// allow only 1 primitive per node as intersecting nodes is cheaper than intersecting bottom level BVHs
		m_Nodes[0].subdivide<7, 32, 1>(transformedAABBs.data(), m_Nodes.data(), m_PrimIndices.data(), 1, m_PoolPtr);
		if (m_PoolPtr > 2)
		{
			m_Nodes[0].bounds.count = -1;
			m_Nodes[0].set_left_first(2);
		}
		else
		{
			m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());
			m_Nodes[0].set_left_first(0);
		}

#if REFIT
		instanceCountChanged = false;
	}
	else // Refit
	{
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.grow(aabb);

		m_Nodes[0].refit(m_Nodes.data(), m_PrimIndices.data(), transformedAABBs.data());
	}
#endif

	if (m_PoolPtr <= 4) // Original tree first in single MBVH node
	{
		m_MNodes.resize(1);
		MBVHNode &mRootNode = m_MNodes[0];
		const AABB invalidAABB = AABB();

		for (int i = 0, s = m_PoolPtr; i < s; i++)
		{
			BVHNode &curNode = m_Nodes[i];

			if (curNode.is_leaf())
			{
				mRootNode.childs[i] = curNode.get_left_first();
				mRootNode.counts[i] = curNode.get_count();
				mRootNode.set_bounds(i, curNode.bounds);
			}
			else
			{
				mRootNode.childs[i] = 0;
				mRootNode.counts[i] = 0;
				mRootNode.set_bounds(i, invalidAABB);
			}
		}

		for (int i = m_PoolPtr; i < 4; i++)
		{
			mRootNode.childs[i] = 0;
			mRootNode.counts[i] = 0;
			mRootNode.set_bounds(i, invalidAABB);
		}
	}
	else
	{
		m_MPoolPtr.store(1);
		m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
		m_MNodes[0].merge_nodes(m_Nodes[0], m_Nodes.data(), m_MNodes.data(), m_MPoolPtr);
	}
}

std::optional<const rfw::Triangle> rfw::TopLevelBVH::intersect(cpurt::Ray &ray, float t_min, uint &instID) const
{
	return intersect(ray.origin, ray.direction, &ray.t, &ray.primIdx, t_min, instID);
}

std::optional<const rfw::Triangle> rfw::TopLevelBVH::intersect(const vec3 &org, const vec3 &dir, float *t, int *primID, float t_min, uint &instID) const
{
	int stackPtr = 0;
	const auto mask = _mm_set_epi32(0, ~0, ~0, ~0);
	const vec3 dirInverse = 1.0f / dir;

	union {
		vec4 origin;
		__m128 org4;
	};
	origin = vec4(org, 1);

	union {
		vec4 direction;
		__m128 dir4;
	};
	direction = vec4(dir, 0);

	vec3 tmp_org;
	vec3 tmp_dir;

#if USE_TOP_MBVH
	MBVHTraversal todo[32];
	todo[0].leftFirst = 0;
	todo[0].count = -1;
	float t_near, t_far;

	while (stackPtr >= 0)
	{
		const int leftFirst = todo[stackPtr].leftFirst;
		const int count = todo[stackPtr].count;
		stackPtr--;

		if (count > -1)
		{
			// leaf node
			for (int i = 0; i < count; i++)
			{
				const auto primIdx = m_PrimIndices[leftFirst + i];
				if (transformedAABBs[primIdx].intersect(org, dirInverse, &t_near, &t_far, t_min))
				{
					const glm_vec4 origin4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, org4);
					const glm_vec4 direction4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, dir4);

					_mm_maskstore_ps(value_ptr(tmp_org), mask, origin4);
					_mm_maskstore_ps(value_ptr(tmp_dir), mask, direction4);

					const CPUMesh *mesh = accelerationStructures[primIdx];
#if USE_MBVH
					if (mesh->mbvh->traverse(tmp_org, tmp_dir, t_min, t, primID))
						instID = primIdx;
#else
					if (mesh->bvh->traverse(tmp_org, tmp_dir, t_min, t, primID))
						instID = primIdx;
#endif
				}
			}
			continue;
		}

		const auto hitInfo = m_MNodes[leftFirst].intersect(origin, dirInverse, t, t_min);
		for (auto i = 3; i >= 0; i--)
		{
			// reversed order, we want to check best nodes first
			const int idx = (hitInfo.tmini[i] & 0b11);
			if (hitInfo.result[idx])
			{
				stackPtr++;
				todo[stackPtr].leftFirst = m_MNodes[leftFirst].childs[idx];
				todo[stackPtr].count = m_MNodes[leftFirst].counts[idx];
			}
		}
	}
#else
	BVHTraversal todo[32];
	float tNear1, tFar1;
	float tNear2, tFar2;
	todo[stackPtr].nodeIdx = 0;

	while (stackPtr >= 0)
	{
		const auto &node = m_Nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const auto primIdx = m_PrimIndices[node.get_left_first() + i];
				if (transformedAABBs[primIdx].intersect(org, dirInverse, &tNear1, &tFar1, t_min))
				{
					const glm_vec4 origin4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, org4);
					const glm_vec4 direction4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, dir4);

					_mm_maskstore_ps(value_ptr(tmp_org), mask, origin4);
					_mm_maskstore_ps(value_ptr(tmp_dir), mask, direction4);

					const CPUMesh *mesh = accelerationStructures[primIdx];
#if USE_MBVH
					if (mesh->mbvh->traverse(tmp_org, tmp_dir, t_min, t, primID))
						instID = primIdx;
#else
					if (mesh->bvh->traverse(tmp_org, tmp_dir, t_min, t, primID))
						instID = primIdx;
#endif
				}
			}
		}
		else
		{
			const auto hit_left = m_Nodes[node.get_left_first()].intersect(org, dirInverse, &tNear1, &tFar1, t_min);
			const auto hit_right = m_Nodes[node.get_left_first() + 1].intersect(org, dirInverse, &tNear2, &tFar2, t_min);

			if (hit_left && hit_right)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
			}
			else if (hit_left)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hit_right)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}
#endif

	if (*t < 1e33f)
	{
		auto tri = accelerationStructures[instID]->triangles[*primID];

		__m128 vertex0 = _mm_load_ps(value_ptr(tri.vertex0));
		__m128 vertex1 = _mm_load_ps(value_ptr(tri.vertex1));
		__m128 vertex2 = _mm_load_ps(value_ptr(tri.vertex2));

		vertex0 = glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex0);
		vertex1 = glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex1);
		vertex2 = glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex2);

		tri.vertex0 = glm::make_vec4(reinterpret_cast<const float *>(&vertex0));
		tri.vertex1 = glm::make_vec4(reinterpret_cast<const float *>(&vertex1));
		tri.vertex2 = glm::make_vec4(reinterpret_cast<const float *>(&vertex2));

		const __m128 vN0 = glm_mat4_mul_vec4(inverseNormalMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN0), mask));
		const __m128 vN1 = glm_mat4_mul_vec4(inverseNormalMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN1), mask));
		const __m128 vN2 = glm_mat4_mul_vec4(inverseNormalMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN2), mask));

		_mm_maskstore_ps(value_ptr(tri.vN0), mask, vN0);
		_mm_maskstore_ps(value_ptr(tri.vN1), mask, vN1);
		_mm_maskstore_ps(value_ptr(tri.vN2), mask, vN2);

		// const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
		const __m128 v1_sub_v0 = glm_vec4_sub(vertex1, vertex0);
		const __m128 v2_sub_v0 = glm_vec4_sub(vertex2, vertex0);
		const __m128 cross_product = glm_vec4_cross(v1_sub_v0, v2_sub_v0);
		const __m128 normalized = glm_vec4_normalize(cross_product);
		const vec3 N = glm::make_vec3(reinterpret_cast<const float *>(&normalized));

		tri.Nx = N.x;
		tri.Ny = N.y;
		tri.Nz = N.z;

		return std::make_optional(tri);
	}

	return std::nullopt;
}

void rfw::TopLevelBVH::intersect(cpurt::RayPacket4 &packet, float t_min) const
{
	int stackPtr = 0;
	cpurt::RayPacket4 transformedPacket = {};

#if USE_TOP_MBVH
	MBVHTraversal todo[32];
	todo[0].leftFirst = 0;
	todo[0].count = -1;

	while (stackPtr >= 0)
	{
		const int leftFirst = todo[stackPtr].leftFirst;
		const int count = todo[stackPtr].count;
		stackPtr--;

		if (count > -1)
		{
			// leaf node
			for (int i = 0; i < count; i++)
			{
				__m128 tNear1, tFar1;
				const auto inst_id = m_PrimIndices[leftFirst + i];
				if (transformedAABBs[inst_id].intersect(packet, &tNear1, &tFar1, t_min))
				{
					const auto &matrix = inverseMatrices[inst_id];

					for (int i = 0; i < 4; i++)
					{
						union {
							__m128 org4;
							float org[4];
						};
						org4 = _mm_setr_ps(packet.origin_x[i], packet.origin_y[i], packet.origin_z[i], 1.0f);
						union {
							__m128 dir4;
							float dir[4];
						};
						dir4 = _mm_setr_ps(packet.direction_x[i], packet.direction_y[i], packet.direction_z[i], 0.0f);

						org4 = glm_mat4_mul_vec4(matrix.cols, org4);
						dir4 = glm_vec4_normalize(glm_mat4_mul_vec4(matrix.cols, dir4));

						transformedPacket.origin_x[i] = org[0];
						transformedPacket.origin_y[i] = org[1];
						transformedPacket.origin_z[i] = org[2];

						transformedPacket.direction_x[i] = dir[0];
						transformedPacket.direction_y[i] = dir[1];
						transformedPacket.direction_z[i] = dir[2];
					}

					__m128 hit_mask = _mm_setzero_ps();
					const auto mesh = accelerationStructures[inst_id];
#if USE_MBVH
					mesh->mbvh->traverse(transformedPacket, t_min, &hit_mask);
#else
					mesh->bvh->traverse(transformedPacket, t_min, &hit_mask);
#endif
					const __m128i storage_mask = _mm_castps_si128(hit_mask);
					_mm_maskstore_ps(packet.t, storage_mask, transformedPacket.t4[0]);
					_mm_maskstore_epi32(packet.instID, storage_mask, _mm_set1_epi32(inst_id));
					_mm_maskstore_epi32(packet.primID, storage_mask, transformedPacket.primID4[0]);
				}
			}
			continue;
		}

		const MBVHHit hit = m_MNodes[leftFirst].intersect4(packet, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackPtr++;
				todo[stackPtr].leftFirst = m_MNodes[leftFirst].childs[idx];
				todo[stackPtr].count = m_MNodes[leftFirst].counts[idx];
			}
		}
	}
#else
	BVHTraversal todo[32];
	__m128 tNear1, tFar1;
	__m128 tNear2, tFar2;
	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = m_Nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const auto inst_id = m_PrimIndices[node.get_left_first() + i];

				if (transformedAABBs[inst_id].intersect(packet, &tNear1, &tFar1, t_min))
				{
					const auto &matrix = inverseMatrices[inst_id];

					for (int i = 0; i < 4; i++)
					{
						union {
							__m128 org4;
							float org[4];
						};
						org4 = _mm_setr_ps(packet.origin_x[i], packet.origin_y[i], packet.origin_z[i], 1.0f);
						union {
							__m128 dir4;
							float dir[4];
						};
						dir4 = _mm_setr_ps(packet.direction_x[i], packet.direction_y[i], packet.direction_z[i], 0.0f);

						org4 = glm_mat4_mul_vec4(matrix.cols, org4);
						dir4 = glm_mat4_mul_vec4(matrix.cols, dir4);
						dir4 = glm_vec4_normalize(dir4);

						transformedPacket.origin_x[i] = org[0];
						transformedPacket.origin_y[i] = org[1];
						transformedPacket.origin_z[i] = org[2];

						transformedPacket.direction_x[i] = dir[0];
						transformedPacket.direction_y[i] = dir[1];
						transformedPacket.direction_z[i] = dir[2];
					}

					__m128 hit_mask = _mm_setzero_ps();
					const auto mesh = accelerationStructures[inst_id];
#if USE_MBVH
					mesh->mbvh->traverse(transformedPacket, t_min, &hit_mask);
#else
					mesh->bvh->traverse(transformedPacket, t_min, &hit_mask);
#endif
					const __m128i storage_mask = _mm_castps_si128(hit_mask);
					_mm_maskstore_ps(packet.t, storage_mask, *reinterpret_cast<__m128 *>(transformedPacket.t));
					_mm_maskstore_epi32(packet.instID, storage_mask, _mm_set1_epi32(inst_id));
					_mm_maskstore_epi32(packet.primID, storage_mask, *reinterpret_cast<__m128i *>(transformedPacket.primID));
				}
			}
		}
		else
		{
			const bool hitLeft = m_Nodes[node.get_left_first()].intersect(packet, &tNear1, &tFar1, t_min);
			const bool hitRight = m_Nodes[node.get_left_first() + 1].intersect(packet, &tNear2, &tFar2, t_min);

			if (hitLeft && hitRight)
			{
				if (_mm_movemask_ps(_mm_cmplt_ps(tNear1, tNear2)) > _mm_movemask_ps(_mm_cmpge_ps(tNear1, tNear2)) /* tNear1 < tNear2*/)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}
#endif
}

void rfw::TopLevelBVH::setInstance(int idx, glm::mat4 transform, CPUMesh *tree, AABB boundingBox)
{
	while (idx >= static_cast<int>(accelerationStructures.size()))
	{
		instanceCountChanged = true;
		transformedAABBs.emplace_back();
		boundingBoxes.emplace_back();
		accelerationStructures.emplace_back();
		instanceMatrices.emplace_back();
		inverseNormalMatrices.emplace_back();
		inverseMatrices.emplace_back();
	}

	boundingBoxes[idx] = boundingBox;
	accelerationStructures[idx] = tree;
	instanceMatrices[idx] = transform;
	inverseMatrices[idx] = inverse(transform);
	inverseNormalMatrices[idx] = mat4(transpose(inverse(mat3(transform))));
	transformedAABBs[idx] = calculateWorldBounds(boundingBox, instanceMatrices[idx]);
}

AABB rfw::TopLevelBVH::calculateWorldBounds(const AABB &originalBounds, const simd::matrix4 &matrix)
{
	const __m128 p1 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f));
	const __m128 p5 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f));

	const __m128 p2 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f));
	const __m128 p3 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f));

	const __m128 p4 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f));
	const __m128 p6 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f));

	const __m128 p7 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f));
	const __m128 p8 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f));

	AABB transformedAABB = {};
	transformedAABB.grow(p1);
	transformedAABB.grow(p2);
	transformedAABB.grow(p3);
	transformedAABB.grow(p4);
	transformedAABB.grow(p5);
	transformedAABB.grow(p6);
	transformedAABB.grow(p7);
	transformedAABB.grow(p8);

	const __m128 epsilon4 = _mm_set1_ps(1e-5f);
	transformedAABB.bmin4 = _mm_sub_ps(transformedAABB.bmin4, epsilon4);
	transformedAABB.bmax4 = _mm_sub_ps(transformedAABB.bmax4, epsilon4);

	return transformedAABB;
}

const rfw::Triangle &rfw::TopLevelBVH::get_triangle(int instID, int primID) const
{
	assert(instID > 0 && instID < accelerationStructures.size());
	return accelerationStructures[instID]->triangles[primID];
}

const rfw::simd::matrix4 &rfw::TopLevelBVH::get_normal_matrix(int instID) const { return inverseNormalMatrices[instID]; }

const rfw::simd::matrix4 &rfw::TopLevelBVH::get_instance_matrix(int instID) const { return instanceMatrices[instID]; }
