#pragma once

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

	BVHTree *bvh;
	std::vector<MBVHNode> mbvh_nodes;
	void construct_bvh(bool printBuildTime = false);

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	bool traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float tmax);

	AABB get_aabb() const { return bvh->aabb; }
	operator bool() const { return !mbvh_nodes.empty(); }

	std::atomic_int pool_ptr = 0;
};

} // namespace bvh
} // namespace rfw