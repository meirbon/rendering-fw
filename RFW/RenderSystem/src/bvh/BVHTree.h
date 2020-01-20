#pragma once

#include "BVHNode.h"
#include "AABB.h"

namespace rfw
{
namespace bvh
{
class BVHTree
{
  public:
	BVHTree(const glm::vec4 *vertices, int vertexCount);
	BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount);

	void construct_bvh(bool printBuildTime = false);
	void reset();

	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);

	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	bool traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max);

	void set_vertices(const glm::vec4 *vertices);
	void set_vertices(const glm::vec4 *vertices, const glm::uvec3 *indices);

	operator bool() const { return !bvh_nodes.empty(); }

  public:
	const glm::vec4 *vertices;
	const glm::uvec3 *indices;

	const int vertex_count;
	const int face_count;

	AABB aabb;
	std::vector<AABB> aabbs;
	std::vector<BVHNode> bvh_nodes;
	std::vector<unsigned int> prim_indices;

	std::vector<glm::vec3> p0s;
	std::vector<glm::vec3> edge1s;
	std::vector<glm::vec3> edge2s;

	std::atomic_int pool_ptr;
	std::atomic_int building_threads = 0;
};
} // namespace bvh
} // namespace rfw