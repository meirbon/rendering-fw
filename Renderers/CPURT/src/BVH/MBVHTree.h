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
	unsigned int m_FinalPtr = 0;
	void constructBVH();
	void traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);

	AABB aabb;

  private:
	std::mutex m_PoolPtrMutex{};
	std::mutex m_ThreadMutex{};
	unsigned int m_BuildingThreads = 0;
	bool m_ThreadLimitReached = false;
};
