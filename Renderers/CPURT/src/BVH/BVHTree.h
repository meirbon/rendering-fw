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

	void constructBVH();
	void buildBVH();
	void reset();

	void setVertices(const glm::vec4 *vertices);

  public:
	const glm::vec4 *m_Vertices;
	const glm::uvec3 *m_Indices;

	const int m_VertexCount;
	const int m_FaceCount;

	AABB aabb;
	std::vector<AABB> m_AABBs;
	std::vector<BVHNode> m_BVHPool;
	std::vector<unsigned int> m_PrimitiveIndices;

	std::atomic_int m_PoolPtr;
	std::mutex m_PoolPtrMutex{};
	std::mutex m_ThreadMutex{};
	unsigned int m_BuildingThreads = 0;
};