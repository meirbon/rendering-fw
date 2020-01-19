#pragma once

#include "../PCH.h"

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

	operator bool() const { return !m_BVHPool.empty(); }

  public:
	const glm::vec4 *m_Vertices;
	const glm::uvec3 *m_Indices;

	const int m_VertexCount;
	const int m_FaceCount;

	AABB aabb;
	std::vector<AABB> m_AABBs;
	std::vector<BVHNode> m_BVHPool;
	std::vector<unsigned int> m_PrimitiveIndices;

	std::vector<glm::vec3> p0s;
	std::vector<glm::vec3> edge1s;
	std::vector<glm::vec3> edge2s;

	std::atomic_int m_PoolPtr;
	std::atomic_int m_BuildingThreads = 0;
};
} // namespace bvh
} // namespace rfw