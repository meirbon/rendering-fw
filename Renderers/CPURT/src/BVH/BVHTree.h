#pragma once

#include <iostream>
#include <vector>
#include <mutex>

#include "BVH/BVHNode.h"
#include "BVH/AABB.h"

class BVHTree
{
  public:
	BVHTree(const glm::vec4 *vertices, int vertexCount);
	BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount);

	void constructBVH(bool printBuildTime = false);
	void reset();
	void refit(const glm::vec4 *vertices);
	void refit(const glm::vec4 *vertices, const glm::uvec3 *indices);
	bool traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx);
	int traverse(cpurt::RayPacket4 &packet, float t_min, __m128* hit_mask);

	void set_vertices(const glm::vec4 *vertices);
	void set_vertices(const glm::vec4 *vertices, const glm::uvec3 *indices);

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