#pragma once

#include <mutex>

#include "BVH/MBVHNode.h"
#include "BVH/BVHTree.h"

class AABB;
class MBVHNode;

class MBVHTree
{
  public:
	friend class MBVHNode;
	explicit MBVHTree(BVHTree *orgTree);

	BVHTree *m_OriginalTree;
	std::vector<MBVHNode> m_Tree;
	std::vector<unsigned int> m_PrimitiveIndices{};
	void constructBVH(bool printBuildTime = false);

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);

	AABB aabb;

  private:
	std::atomic_int m_BuildingThreads = 0;
	std::atomic_int m_PoolPtr = 0;
	bool m_ThreadLimitReached = false;
};
