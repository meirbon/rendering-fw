#pragma once

#include "../PCH.h"

namespace rfw
{
class TopLevelBVH
{
  public:
	TopLevelBVH() = default;

	void construct_bvh();
	void refit();

	// (Optionally) returns hit triangle
	std::optional<const rfw::Triangle> intersect(cpurt::Ray &ray, float t_min, int &instID) const;
	std::optional<const rfw::Triangle> intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID, float t_min, int &instID) const;
	bool rfw::TopLevelBVH::occluded(const vec3 &origin, const vec3 &direction, float t_min, float t_max) const;

	void intersect(cpurt::RayPacket4 &packet, float t_min) const;

	CPUMesh &get_mesh(const int ID) { return *accelerationStructures[ID]; }

	void set_instance(int idx, glm::mat4 transform, CPUMesh *tree, AABB boundingBox);

	static AABB calculate_world_bounds(const AABB &originalBounds, const simd::matrix4 &matrix);

	const Triangle &get_triangle(int instID, int primID) const;
	const simd::matrix4 &get_normal_matrix(int instID) const;
	const simd::matrix4 &get_instance_matrix(int instID) const;

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
	std::vector<simd::matrix4> instanceMatrices;
	std::vector<simd::matrix4> inverseMatrices;
	std::vector<simd::matrix4> inverseNormalMatrices;

	bool instanceCountChanged = true;
};

} // namespace rfw
