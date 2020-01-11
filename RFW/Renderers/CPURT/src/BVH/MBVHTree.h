#pragma once

#include "../PCH.h"

class AABB;
class MBVHNode;

class MBVHTree
{
  public:
	friend class MBVHNode;
	explicit MBVHTree(BVHTree *orgTree);

	BVHTree *m_OriginalTree;
	std::vector<MBVHNode> m_Tree;
	void construct_bvh(bool printBuildTime = false);

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	int traverse(cpurt::RayPacket4 &packet, float t_min, __m128 *hit_mask);

	AABB get_aabb() const { return m_OriginalTree->aabb; }

  private:
	std::atomic_int m_BuildingThreads = 0;
	std::atomic_int m_PoolPtr = 0;
	bool m_ThreadLimitReached = false;
};
