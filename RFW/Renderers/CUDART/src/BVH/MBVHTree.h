#pragma once

#include "../PCH.h"

namespace rfw
{
namespace bvh
{

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
	bool traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float tmax);

	AABB get_aabb() const { return m_OriginalTree->aabb; }

	operator bool() const { return !m_Tree.empty(); }

  private:
	std::atomic_int m_BuildingThreads = 0;
	std::atomic_int m_PoolPtr = 0;
	bool m_ThreadLimitReached = false;
};

} // namespace bvh
} // namespace rfw