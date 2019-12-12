#include "BVH/BVHTree.h"
#include "utils/Timer.h"
#include "../Triangle.h"

#define PRINT_BUILD_TIME 1

using namespace glm;
using namespace rfw;

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount) : m_VertexCount(vertexCount), m_FaceCount(vertexCount / 3)
{
	m_Indices = nullptr;
	m_PoolPtr.store(0);
	reset();
	setVertices(vertices);
}

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount) : m_VertexCount(vertexCount), m_FaceCount(faceCount)
{
	m_Indices = indices;
	m_PoolPtr.store(0);
	reset();
	setVertices(vertices);
}

void BVHTree::constructBVH()
{
	assert(m_Vertices);
	buildBVH();
}

void BVHTree::buildBVH()
{
	if (m_FaceCount > 0)
	{
#if PRINT_BUILD_TIME
		rfw::utils::Timer t;
		t.reset();
#endif
		m_PoolPtr = 2;
		m_BVHPool.emplace_back();
		m_BVHPool.emplace_back();

		auto &rootNode = m_BVHPool[0];
		rootNode.bounds.leftFirst = 0;
		rootNode.bounds.count = static_cast<int>(m_FaceCount);
		rootNode.CalculateBounds(m_AABBs.data(), m_PrimitiveIndices.data());

		rootNode.SubdivideMT(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), &m_ThreadMutex, &m_PoolPtrMutex, &m_BuildingThreads, 1, m_PoolPtr);

		if (m_PoolPtr > 2)
			rootNode.bounds.count = -1, rootNode.SetLeftFirst(2);
		else
			rootNode.bounds.count = static_cast<int>(m_FaceCount);

		m_BVHPool.resize(m_PoolPtr);
#if PRINT_BUILD_TIME
		std::cout << "Building BVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr << std::endl;
#endif
	}
}

void BVHTree::reset()
{
	m_BVHPool.clear();
	if (m_FaceCount > 0)
	{
		m_PrimitiveIndices.clear();
		m_PrimitiveIndices.reserve(m_FaceCount);
		for (int i = 0; i < m_FaceCount; i++)
			m_PrimitiveIndices.push_back(i);
		m_BVHPool.resize(m_FaceCount * 2);
	}
}

void BVHTree::setVertices(const glm::vec4 *vertices)
{
	m_Vertices = vertices;
	aabb = AABB();

	m_AABBs.resize(m_FaceCount);
	if (m_Indices)
	{
		for (int i = 0; i < m_FaceCount; i++)
		{
			const uvec3 &idx = m_Indices[i];
			m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
			aabb.Grow(m_AABBs[i]);
		}
	}
	else
	{
		for (int i = 0; i < m_FaceCount; i++)
		{
			const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);
			m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
			aabb.Grow(m_AABBs[i]);
		}
	}
}
