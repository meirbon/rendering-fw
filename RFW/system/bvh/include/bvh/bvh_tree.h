#pragma once

#include "bvh_node.h"
#include "aabb.h"
#include <rtbvh.hpp>

#include <vector>
#include <optional>

namespace rfw
{
namespace bvh
{
class BVHTree
{
  public:
	enum Type
	{
		BinnedSAH,
		LocallyOrderedClustering,
		SpatialSAH
	};

	BVHTree();
	BVHTree(const glm::vec4 *vertices, int vertexCount);
	BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount);
	~BVHTree();

	void reset();
	void construct(Type type = BinnedSAH);

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx, glm::vec2 *bary);
	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	int traverse4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
				  const float dir_y[4], const float dir_z[4], float t[4], int primID[4], float t_min, __m128 *hit_mask);
	bool traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max);

	void set_vertices(const glm::vec4 *vertices);
	void set_vertices(const glm::vec4 *vertices, const glm::uvec3 *indices);

	AABB get_aabb() const;

	operator bool() const { return instance.has_value(); }

  public:
	const glm::vec4 *vertices = nullptr;
	const glm::uvec3 *indices = nullptr;

	const int vertex_count = -1;
	const int face_count = -1;

	std::optional<rtbvh::RTBVH> instance = std::nullopt;
	std::vector<AABB> aabbs;
	std::vector<glm::vec4> splat_vertices;

	std::vector<glm::vec3> p0s;
	std::vector<glm::vec3> edge1s;
	std::vector<glm::vec3> edge2s;
};
} // namespace bvh
} // namespace rfw