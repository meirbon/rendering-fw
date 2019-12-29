#define GLM_FORCE_AVX
#include "TopLevelBVH.h"

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

	m_MPoolPtr.store(1);
	m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
	m_MNodes[0].MergeNodes(m_Nodes[0], m_Nodes.data(), m_MNodes.data(), m_MPoolPtr);
}

std::optional<const rfw::Triangle> rfw::TopLevelBVH::intersect(Ray &ray, float t_min) const
{
	int stackPtr = 0;
	int instIdx = -1;
	const vec3 dirInverse = 1.0f / ray.direction;

	const vec4 org = vec4(ray.origin, 1);

#if 0
	BVHTraversal todo[32];
	float tNear1, tFar1;
	float tNear2, tFar2;
	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = m_Nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.GetCount() > -1)
		{
			for (int i = 0; i < node.GetCount(); i++)
			{
				const auto primIdx = m_PrimIndices[node.GetLeftFirst() + i];

				if (transformedAABBs[primIdx].Intersect(ray.origin, dirInverse, &tNear1, &tFar1) && tNear1 < ray.t)
				{
					// Transform ray to local space for mesh BVH
					const auto &matrix = instanceMatrices[primIdx];
					const vec3 origin = matrix * vec4(ray.origin, 1);
					const vec3 direction = matrix * vec4(ray.direction, 0);
					const auto mesh = accelerationStructures[primIdx];
					if (mesh->mbvh->traverse(origin, direction, t_min, &ray.t, &ray.primIdx))
						instIdx = primIdx;
				}
			}
		}
		else
		{
			bool hitLeft = m_Nodes[node.GetLeftFirst()].Intersect(ray.origin, dirInverse, &tNear1, &tFar1);
			bool hitRight = m_Nodes[node.GetLeftFirst() + 1].Intersect(ray.origin, dirInverse, &tNear2, &tFar2);

			if (hitLeft && hitRight)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst(), tNear1};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
			}
		}
	}
#else
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
			for (auto i = 0; i < count; i++)
			{
				const auto primIdx = m_PrimIndices[leftFirst + i];
				if (transformedAABBs[primIdx].Intersect(ray.origin, dirInverse, &t_near, &t_far))
				{
					// Transform ray to local space for mesh BVH
					const auto origin = instanceMatrices[primIdx] * org;
					const auto direction = instanceMatrices3[primIdx] * ray.direction;
					const auto &mesh = accelerationStructures[primIdx];
					if (mesh->mbvh->traverse(origin, direction, t_min, &ray.t, &ray.primIdx))
						instIdx = primIdx;
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

#endif

	if (ray.isValid())
	{
		auto tri = accelerationStructures[instIdx]->triangles[ray.primIdx];
		tri.vertex0 = instanceMatrices[instIdx] * vec4(tri.vertex0, 1.0f);
		tri.vertex1 = instanceMatrices[instIdx] * vec4(tri.vertex1, 1.0f);
		tri.vertex2 = instanceMatrices[instIdx] * vec4(tri.vertex2, 1.0f);

		tri.vN0 = inverseNormalMatrices[instIdx] * tri.vN0;
		tri.vN1 = inverseNormalMatrices[instIdx] * tri.vN1;
		tri.vN2 = inverseNormalMatrices[instIdx] * tri.vN2;

		const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
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
		transformedAABBs.emplace_back(calculateWorldBounds(boundingBox, transform));
		boundingBoxes.emplace_back(boundingBox);
		accelerationStructures.push_back(tree);
		instanceMatrices.push_back(transform);
		instanceMatrices3.emplace_back(transform);
		inverseNormalMatrices.emplace_back(transpose(inverse(transform)));
	}
	else
	{
		boundingBoxes[idx] = boundingBox;
		transformedAABBs[idx] = calculateWorldBounds(boundingBox, inverse(transform));
		accelerationStructures[idx] = tree;
		instanceMatrices[idx] = transform;
		instanceMatrices3[idx] = mat3(transform);
		inverseNormalMatrices[idx] = transpose(inverse(transform));
	}
}

AABB rfw::TopLevelBVH::calculateWorldBounds(const AABB &originalBounds, const glm::mat4 &matrix)
{
	const auto transform = transpose(inverse(matrix));

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
