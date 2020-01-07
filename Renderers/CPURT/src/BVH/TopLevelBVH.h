#pragma once

#include "../Mesh.h"
#include "MBVHTree.h"
#include "BVHTree.h"
#include "../Ray.h"

#include <Structures.h>

#include <vector>
#include <tuple>
#include <optional>

namespace rfw
{
class TopLevelBVH
{
  public:
	TopLevelBVH() = default;

	void constructBVH();

	// (Optionally) returns hit triangle
	std::optional<const rfw::Triangle> intersect(cpurt::Ray &ray, float t_min, uint &instID) const;

	std::optional<const rfw::Triangle> intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID, float t_min, uint &instID) const;

	void intersect(cpurt::RayPacket4 &packet, float t_min) const;

	CPUMesh *getMesh(const int ID) { return accelerationStructures[ID]; }

	void setInstance(int idx, glm::mat4 transform, CPUMesh *tree, AABB boundingBox);

	static AABB calculateWorldBounds(const AABB &originalBounds, const simd::matrix4 &matrix);

	const Triangle &get_triangle(int instID, int primID) const;
	const simd::matrix4 &get_normal_matrix(int instID) const;
	const simd::matrix4 &get_instance_matrix(int instID) const;

	private :
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
