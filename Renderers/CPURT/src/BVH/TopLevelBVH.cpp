#define GLM_FORCE_AVX
#include "TopLevelBVH.h"

#define USE_MBVH 1

void rfw::TopLevelBVH::constructBVH()
{
	if (instanceCountChanged) // (Re)build
	{
		m_Nodes.clear();
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.Grow(aabb);

		m_Nodes.resize(boundingBoxes.size() * 2);
		m_PrimIndices.resize(boundingBoxes.size());
		for (uint i = 0, s = static_cast<uint>(m_PrimIndices.size()); i < s; i++)
			m_PrimIndices[i] = i;

		m_PoolPtr.store(2);
		m_Nodes[0].bounds = rootBounds;
		m_Nodes[0].bounds.leftFirst = 0;
		m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());
		m_Nodes[0].Subdivide(transformedAABBs.data(), m_Nodes.data(), m_PrimIndices.data(), 1, m_PoolPtr);
		if (m_PoolPtr > 2)
		{
			m_Nodes[0].bounds.count = -1;
			m_Nodes[0].SetLeftFirst(2);
		}
		else
		{
			m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());
			m_Nodes[0].SetLeftFirst(0);
		}

		if (m_PoolPtr <= 4) // Original tree first in single MBVH node
		{
			m_MNodes.resize(1);
			MBVHNode &mRootNode = m_MNodes[0];
			const AABB invalidAABB = AABB();

			for (int i = 0, s = m_PoolPtr; i < s; i++)
			{
				BVHNode &curNode = m_Nodes[i];

				if (curNode.IsLeaf())
				{
					mRootNode.childs[i] = curNode.GetLeftFirst();
					mRootNode.counts[i] = curNode.GetCount();
					mRootNode.SetBounds(i, curNode.bounds);
				}
				else
				{
					mRootNode.childs[i] = 0;
					mRootNode.counts[i] = 0;
					mRootNode.SetBounds(i, invalidAABB);
				}
			}

			for (int i = m_PoolPtr; i < 4; i++)
			{
				mRootNode.childs[i] = 0;
				mRootNode.counts[i] = 0;
				mRootNode.SetBounds(i, invalidAABB);
			}
		}
		else
		{
			m_MPoolPtr.store(1);
			m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
			m_MNodes[0].MergeNodes(m_Nodes[0], m_Nodes.data(), m_MNodes.data(), m_MPoolPtr);
		}

		instanceCountChanged = false;
	}
	else // Refit
	{
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.Grow(aabb);

		for (int i = static_cast<int>(m_MNodes.size()) - 1; i >= 0; i--)
		{
			auto &node = m_MNodes[i];
			for (int j = 0; j < 4; j++)
			{
				if (node.counts[j] == 0)
					continue;

				if (node.counts[j] >= 0 || node.childs[j] < i /* Child node cannot be at an earlier index */) // Calculate new bounds of leaf nodes
				{
					auto aabb = AABB(vec3(1e34f), vec3(-1e34f));
					for (int k = node.childs[j], s = node.childs[j] + node.counts[j]; k < s; k++)
						aabb.Grow(transformedAABBs[m_PrimIndices[k]]);

					node.bminx[j] = aabb.bmin[0];
					node.bminy[j] = aabb.bmin[1];
					node.bminz[j] = aabb.bmin[2];

					node.bmaxx[j] = aabb.bmax[0];
					node.bmaxy[j] = aabb.bmax[1];
					node.bmaxz[j] = aabb.bmax[2];
				}
				else // Calculate new bounds of bvh nodes
				{
					const auto &childNode = m_MNodes[node.childs[j]];
					node.bminx[j] = min(childNode.bminx[0], min(childNode.bminx[1], min(childNode.bminx[2], childNode.bminx[3]))) - 1e-5f;
					node.bminy[j] = min(childNode.bminy[0], min(childNode.bminy[1], min(childNode.bminy[2], childNode.bminy[3]))) - 1e-5f;
					node.bminz[j] = min(childNode.bminz[0], min(childNode.bminz[1], min(childNode.bminz[2], childNode.bminz[3]))) - 1e-5f;

					node.bmaxx[j] = max(childNode.bmaxx[0], max(childNode.bmaxx[1], max(childNode.bmaxx[2], childNode.bmaxx[3]))) + 1e-5f;
					node.bmaxy[j] = max(childNode.bmaxy[0], max(childNode.bmaxy[1], max(childNode.bmaxy[2], childNode.bmaxy[3]))) + 1e-5f;
					node.bmaxz[j] = max(childNode.bmaxz[0], max(childNode.bmaxz[1], max(childNode.bmaxz[2], childNode.bmaxz[3]))) + 1e-5f;
				}
			}
		}
	}
}

std::optional<const rfw::Triangle> rfw::TopLevelBVH::intersect(Ray &ray, float t_min, uint &instID) const
{
	const auto mask = _mm_set_epi32(0, ~0, ~0, ~0);

	int stackPtr = 0;
	const vec3 dirInverse = 1.0f / ray.direction;

	MBVHTraversal todo[32];
	todo[0].leftFirst = 0;
	todo[0].count = -1;
	float t_near, t_far;

	union {
		vec4 origin;
		__m128 org4;
	};
	origin = vec4(ray.origin, 1);

	union {
		vec4 direction;
		__m128 dir4;
	};
	direction = vec4(ray.direction, 0);

	vec3 org;
	vec3 dir;

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
				if (transformedAABBs[primIdx].Intersect(ray.origin, dirInverse, &t_near, &t_far))
				{
					const glm_vec4 origin4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, org4);
					const glm_vec4 direction4 = glm_mat4_mul_vec4(inverseMatrices[primIdx].cols, dir4);

					_mm_maskstore_ps(value_ptr(org), mask, origin4);
					_mm_maskstore_ps(value_ptr(dir), mask, direction4);

					const CPUMesh *mesh = accelerationStructures[primIdx];
#if USE_MBVH
					if (mesh->mbvh->traverse(org, dir, t_min, &ray.t, &ray.primIdx))
						instID = primIdx;
#else
					if (mesh->bvh->traverse(org, dir, t_min, &ray.t, &ray.primIdx))
						instID = primIdx;
#endif
				}
			}
			continue;
		}

		const auto hitInfo = m_MNodes[leftFirst].intersect(ray.origin, dirInverse, &ray.t);
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

	if (ray.isValid())
	{
		auto tri = accelerationStructures[instID]->triangles[ray.primIdx];

		const __m128 vertex0 = _mm_load_ps(value_ptr(tri.vertex0));
		const __m128 vertex1 = _mm_load_ps(value_ptr(tri.vertex1));
		const __m128 vertex2 = _mm_load_ps(value_ptr(tri.vertex2));

		tri.vertex0 = glm::make_vec4(reinterpret_cast<const float *>(&glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex0)));
		tri.vertex1 = glm::make_vec4(reinterpret_cast<const float *>(&glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex1)));
		tri.vertex2 = glm::make_vec4(reinterpret_cast<const float *>(&glm_mat4_mul_vec4(instanceMatrices[instID].cols, vertex2)));

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
	inverseNormalMatrices[idx] = transpose(inverse(transform));
	transformedAABBs[idx] = calculateWorldBounds(boundingBox, instanceMatrices[idx]);
}

AABB rfw::TopLevelBVH::calculateWorldBounds(const AABB &originalBounds, const SIMDMat4 &matrix)
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
	transformedAABB.Grow(p1);
	transformedAABB.Grow(p2);
	transformedAABB.Grow(p3);
	transformedAABB.Grow(p4);
	transformedAABB.Grow(p5);
	transformedAABB.Grow(p6);
	transformedAABB.Grow(p7);
	transformedAABB.Grow(p8);

	const __m128 epsilon4 = _mm_set1_ps(1e-5f);
	transformedAABB.bmin4 = _mm_sub_ps(transformedAABB.bmin4, epsilon4);
	transformedAABB.bmax4 = _mm_sub_ps(transformedAABB.bmax4, epsilon4);

	return transformedAABB;
}
