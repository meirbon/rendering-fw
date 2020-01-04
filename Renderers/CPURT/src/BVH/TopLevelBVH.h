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
	std::optional<const rfw::Triangle> intersect(Ray &ray, float t_min, uint &instID) const;

	CPUMesh *getMesh(const int ID) { return accelerationStructures[ID]; }

	void setInstance(int idx, glm::mat4 transform, CPUMesh *tree, AABB boundingBox);

	static AABB calculateWorldBounds(const AABB &originalBounds, const SIMDMat4 &matrix);

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
	std::vector<SIMDMat4> instanceMatrices;
	std::vector<SIMDMat4> inverseMatrices;
	std::vector<SIMDMat4> inverseNormalMatrices;

	bool instanceCountChanged = true;
};

} // namespace rfw
