#pragma once

#include "AABB.h"
#include "BVHTree.h"
#include <rtbvh.hpp>

namespace rfw
{
namespace bvh
{

struct AABB;
class MBVHNode;

class MBVHTree
{
  public:
	friend class MBVHNode;
	MBVHTree();
	MBVHTree(BVHTree *orgTree);
	~MBVHTree();

	void reset();
	void construct();

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx, glm::vec2 *bary);
	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	int traverse4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
				  const float dir_y[4], const float dir_z[4], float t[4], int primID[4], float t_min, __m128 *hit_mask);
	bool traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float tmax);

	AABB get_aabb() const;

	operator bool() const { return instance.has_value(); }

	BVHTree *bvh = nullptr;
	std::optional<rtbvh::RTMBVH> instance = std::nullopt;
};

} // namespace bvh
} // namespace rfw