#pragma once

#include "Mesh.h"
#include "MBVHTree.h"
#include "BVHTree.h"
#include "Ray.h"

#include <vector>

namespace rfw
{
class TopLevelBVH
{
  public:
	TopLevelBVH() = default;

	void constructBVH()
	{
		m_Nodes.clear();
		AABB rootBounds = {};
		for (const auto &aabb : boundingBoxes)
			rootBounds.Grow(aabb);
		transformedAABBs = boundingBoxes;
		for (int i = 0, s = static_cast<int>(transformedAABBs.size()); i < s; i++)
		{
			const auto &matrix = instanceMatrices[i];
			const glm::vec3 minBounds = matrix * glm::vec4(glm::make_vec3(boundingBoxes[i].bmin), 1.0f);
			const glm::vec3 maxBounds = matrix * glm::vec4(glm::make_vec3(boundingBoxes[i].bmax), 1.0f);
			transformedAABBs[i] = AABB(minBounds, maxBounds);
		}

		m_Nodes.resize(boundingBoxes.size() * 2);
		m_PrimIndices.resize(boundingBoxes.size());
		for (uint i = 0, s = static_cast<uint>(m_PrimIndices.size()); i < s; i++)
			m_PrimIndices[i] = i;

		m_PoolPtr.store(2);
		m_Nodes[0].bounds = rootBounds;
		m_Nodes[0].bounds.leftFirst = 0;
		m_Nodes[0].bounds.count = transformedAABBs.size();
		m_Nodes[0].Subdivide(transformedAABBs.data(), m_Nodes.data(), m_PrimIndices.data(), 1, m_PoolPtr);

		m_MPoolPtr.store(1);
		m_MNodes.resize(m_Nodes.size());
		m_MNodes[0].MergeNodes(m_Nodes[0], m_Nodes.data(), m_MNodes.data(), m_MPoolPtr);
	}

	rfw::Triangle *intersect(Ray &ray, float t_min = 0.0f) const
	{
		BVHTraversal todo[32];
		int stackPtr = 0;
		int instIdx = -1;
		float tNear1, tFar1;
		float tNear2, tFar2;

		const glm::vec3 dirInverse = 1.0f / ray.direction;

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

					if (boundingBoxes[primIdx].Intersect(ray.origin, dirInverse, &tNear1, &tFar1) && tNear1 < ray.t)
					{
						// Transform ray to local space for mesh BVH
						const auto &matrix = instanceMatrices[primIdx];
												const vec3 origin = matrix * vec4(ray.origin, 1);
												const vec3 direction = matrix * vec4(ray.direction, 0);

//						const vec3 origin = ray.origin;
//						const vec3 direction = ray.direction;

						const auto mesh = accelerationStructures[primIdx];
						int rayPrimIdx = ray.primIdx;
						mesh->mbvh->traverse(origin, direction, t_min, &ray.t, &rayPrimIdx);
						if (ray.primIdx != rayPrimIdx)
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

		if (ray.isValid())
			return &accelerationStructures[instIdx]->triangles[ray.primIdx];
		else
			return nullptr;
	}

	CPUMesh *getMesh(const int ID) { return accelerationStructures[ID]; }

	void setInstance(int idx, glm::mat4 transform, CPUMesh *tree, AABB boundingBox)
	{
		// Transform AABB to correct position
		auto min = glm::make_vec3(boundingBox.bmin);
		auto max = glm::make_vec3(boundingBox.bmax);
		min = transform * vec4(min, 1);
		max = transform * vec4(max, 1);
		boundingBox = AABB(min, max);

		if (idx >= accelerationStructures.size())
		{
			boundingBoxes.emplace_back(boundingBox);
			accelerationStructures.push_back(tree);
			instanceMatrices.push_back(transform);
			inverseMatrices.push_back(inverse(transform));
			inverseNormalMatrices.emplace_back(transpose(inverse(transform)));
		}
		else
		{
			boundingBoxes[idx] = boundingBox;
			accelerationStructures[idx] = tree;
			instanceMatrices[idx] = transform;
			inverseMatrices[idx] = inverse(transform);
			inverseNormalMatrices[idx] = transpose(inverse(transform));
		}
	}

  private:
	// Top level BVH structure data
	std::atomic_int m_PoolPtr = 0;
	std::atomic_int m_MPoolPtr = 0;
	std::atomic_int m_MThreadCount = 0;
	std::vector<BVHNode> m_Nodes;
	std::vector<MBVHNode> m_MNodes;
	std::vector<AABB> boundingBoxes;
	std::vector<AABB> transformedAABBs;
	std::vector<unsigned int> m_PrimIndices;

	// Instance data
	std::vector<CPUMesh *> accelerationStructures;
	std::vector<glm::mat4> instanceMatrices;
	std::vector<glm::mat4> inverseMatrices;
	std::vector<glm::mat3> inverseNormalMatrices;
};

} // namespace rfw
