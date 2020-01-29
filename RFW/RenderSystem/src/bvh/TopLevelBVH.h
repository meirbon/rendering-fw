#pragma once

namespace rfw
{
namespace bvh
{
class BVHTree;
class MBVHTree;

struct rfwMesh
{
	rfwMesh() = default;

	void set_geometry(const Mesh &mesh);

	BVHTree *bvh = nullptr;
	MBVHTree *mbvh = nullptr;

	const rfw::Triangle *triangles = nullptr;
	const glm::vec4 *vertices = nullptr;
	const glm::uvec3 *indices = nullptr;

	int vertexCount;
	int triangleCount;
};

class TopLevelBVH
{
  public:
	TopLevelBVH() = default;

	void construct_bvh();
	void refit();

	rfwMesh &get_mesh(const int ID) { return *instance_meshes[ID]; }

	const rfw::Triangle *intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID, int *instID,
								   glm::vec2 *bary, float t_min = 1e-5f) const;
	const rfw::Triangle *intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID, int *instID,
								   float t_min = 1e-5f) const;

	int intersect4(float origin_x[4], float origin_y[4], float origin_z[4], float dir_x[4], float dir_y[4],
				   float dir_z[4], float t[4], int primID[4], int instID[4], float t_min) const;
	int intersect4(float origin_x[4], float origin_y[4], float origin_z[4], float dir_x[4], float dir_y[4],
				   float dir_z[4], float t[4], float bary_x[4], float bary_y[4], int primID[4], int instID[4],
				   float t_min) const;

	bool is_occluded(const vec3 &origin, const vec3 &direction, float t_max, float t_min = 1e-5f) const;

	void set_instance(size_t idx, glm::mat4 transform, rfwMesh *tree, AABB boundingBox);

	static AABB calculate_world_bounds(const AABB &originalBounds, const simd::matrix4 &matrix);

	const Triangle &get_triangle(int instID, int primID) const;
	const simd::matrix4 &get_normal_matrix(int instID) const;
	const simd::matrix4 &get_instance_matrix(int instID) const;

	bool count_changed = true;
	// Top level BVH structure data
	std::atomic_int pool_ptr = 0;
	std::atomic_int mpool_ptr = 0;
	std::atomic_int thread_count = 0;
	std::vector<BVHNode> bvh_nodes;
	std::vector<MBVHNode> mbvh_nodes;
	std::vector<AABB> aabbs;
	std::vector<AABB> instance_aabbs;
	std::vector<uint> prim_indices;

	// Instance data
	std::vector<rfwMesh *> instance_meshes;
	std::vector<simd::matrix4> matrices;
	std::vector<simd::matrix4> inverse_matrices;
	std::vector<simd::matrix4> normal_matrices;
};

} // namespace bvh
} // namespace rfw
