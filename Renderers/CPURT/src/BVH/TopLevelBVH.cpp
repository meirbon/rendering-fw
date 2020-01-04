#define GLM_FORCE_AVX
#include "TopLevelBVH.h"

#define USE_MBVH 1

void rfw::TopLevelBVH::constructBVH()
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
				const AABB invalidAABB = {glm::vec3(1e34f), glm::vec3(-1e34f)};
				mRootNode.SetBounds(i, invalidAABB);
			}
		}

		for (int i = m_PoolPtr; i < 4; i++)
		{
			mRootNode.childs[i] = 0;
			mRootNode.counts[i] = 0;
			const AABB invalidAABB = {glm::vec3(1e34f), glm::vec3(-1e34f)};
			mRootNode.SetBounds(i, invalidAABB);
		}
	}
	else
	{
		m_MPoolPtr.store(1);
		m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
		m_MNodes[0].MergeNodes(m_Nodes[0], m_Nodes.data(), m_MNodes.data(), m_MPoolPtr);
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
			if (hitInfo.result[idx] == 1)
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

		const __m128 vN0 = glm_mat4_mul_vec4(inverseMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN0), mask));
		const __m128 vN1 = glm_mat4_mul_vec4(inverseMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN1), mask));
		const __m128 vN2 = glm_mat4_mul_vec4(inverseMatrices[instID].cols, _mm_maskload_ps(value_ptr(tri.vN2), mask));

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
	if (idx >= static_cast<int>(accelerationStructures.size()))
	{
		transformedAABBs.emplace_back();
		boundingBoxes.emplace_back();
		accelerationStructures.emplace_back();
		instanceMatrices.emplace_back();
		instanceMatrices3.emplace_back();
		inverseMatrices.emplace_back();
		inverseMatrices3.emplace_back();
	}

	boundingBoxes[idx] = boundingBox;
	transformedAABBs[idx] = calculateWorldBounds(boundingBox, transform);
	accelerationStructures[idx] = tree;
	instanceMatrices[idx] = transform;
	instanceMatrices3[idx] = mat3(transform);
	inverseMatrices[idx] = inverse(transform);
	inverseMatrices3[idx] = mat3(transpose(inverse(transform)));
}

AABB rfw::TopLevelBVH::calculateWorldBounds(const AABB &originalBounds, const glm::mat4 &matrix)
{
	const auto transform = matrix;

	const vec4 p1 = vec4(glm::make_vec3(originalBounds.bmin), 1.f);
	const vec4 p5 = vec4(glm::make_vec3(originalBounds.bmax), 1.f);

	const vec4 p2 = transform * vec4(p1.x, p1.y, p5.z, 1.f);
	const vec4 p3 = transform * vec4(p1.x, p5.y, p1.z, 1.f);
	const vec4 p4 = transform * vec4(p5.x, p1.y, p1.z, 1.f);

	const vec4 p6 = transform * vec4(p5.x, p5.y, p1.z, 1.f);
	const vec4 p7 = transform * vec4(p5.x, p1.y, p5.z, 1.f);
	const vec4 p8 = transform * vec4(p1.x, p5.y, p5.z, 1.f);

	AABB transformedAABB = {};
	transformedAABB.Grow(p1);
	transformedAABB.Grow(p2);
	transformedAABB.Grow(p3);
	transformedAABB.Grow(p4);
	transformedAABB.Grow(p5);
	transformedAABB.Grow(p6);
	transformedAABB.Grow(p7);
	transformedAABB.Grow(p8);
	return transformedAABB;
}
