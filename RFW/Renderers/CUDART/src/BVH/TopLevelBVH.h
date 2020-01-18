#pragma once

#include "../PCH.h"

namespace rfw
{
namespace bvh
{
class BVHTree;
class MBVHTree;

struct BVHMesh
{
	BVHMesh() = default;

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

	// (Optionally) returns hit triangle

	BVHMesh &get_mesh(const int ID) { return *accelerationStructures[ID]; }

	void set_instance(int idx, glm::mat4 transform, BVHMesh *tree, AABB boundingBox);

	static AABB calculate_world_bounds(const AABB &originalBounds, const simd::matrix4 &matrix);

	const Triangle &get_triangle(int instID, int primID) const;
	const simd::matrix4 &get_normal_matrix(int instID) const;
	const simd::matrix4 &get_instance_matrix(int instID) const;

  public:
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
	std::vector<BVHMesh *> accelerationStructures;
	std::vector<simd::matrix4> instanceMatrices;
	std::vector<simd::matrix4> inverseMatrices;
	std::vector<simd::matrix4> inverseNormalMatrices;

	bool instanceCountChanged = true;
};

} // namespace bvh
} // namespace rfw
