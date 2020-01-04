#include "BVH/BVHTree.h"
#include "utils/Timer.h"
#include "../Triangle.h"

using namespace glm;
using namespace rfw;

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount) : m_VertexCount(vertexCount), m_FaceCount(vertexCount / 3)
{
	m_Indices = nullptr;
	m_PoolPtr.store(0);
	reset();
	m_Vertices = vertices;

	aabb = AABB(vec3(1e34f), vec3(-1e34f));

	for (int i = 0; i < m_VertexCount; i++)
		aabb.Grow(m_Vertices[i]);

	m_AABBs.resize(m_FaceCount);
	for (int i = 0; i < m_FaceCount; i++)
	{
		const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	}
}

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount) : m_VertexCount(vertexCount), m_FaceCount(faceCount)
{
	m_PoolPtr.store(0);
	reset();
	m_Vertices = vertices;
	m_Indices = indices;

	aabb = AABB(vec3(1e34f), vec3(-1e34f));

	for (int i = 0; i < m_VertexCount; i++)
		aabb.Grow(m_Vertices[i]);

	m_AABBs.resize(m_FaceCount);
	for (int i = 0; i < m_FaceCount; i++)
	{
		const uvec3 &idx = m_Indices[i];
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	}
}

void BVHTree::constructBVH(bool printBuildTime)
{
	assert(m_Vertices);

	if (m_FaceCount > 0)
	{
		utils::Timer t = {};
		m_PoolPtr = 2;
		m_BVHPool.emplace_back();
		m_BVHPool.emplace_back();

		auto &rootNode = m_BVHPool[0];
		rootNode.bounds.leftFirst = 0;
		rootNode.bounds.count = static_cast<int>(m_FaceCount);
		rootNode.CalculateBounds(m_AABBs.data(), m_PrimitiveIndices.data());

		// rootNode.Subdivide(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), 1, m_PoolPtr);
		rootNode.SubdivideMT(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), &m_ThreadMutex, &m_BuildingThreads, 1, m_PoolPtr);

		if (m_PoolPtr > 2)
		{
			rootNode.bounds.count = -1;
			rootNode.SetLeftFirst(2);
		}
		else
		{
			rootNode.bounds.count = static_cast<int>(m_FaceCount);
			rootNode.SetLeftFirst(0);
		}

		m_BVHPool.resize(m_PoolPtr);

		if (printBuildTime)
			std::cout << "Building BVH took: " << t.elapsed() << " ms for " << m_FaceCount << " triangles. Poolptr: " << m_PoolPtr << std::endl;
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

void BVHTree::refit(const glm::vec4 *vertices)
{
	m_Vertices = vertices;

	aabb = AABB();

	// Recalculate AABBs
	m_AABBs.resize(m_FaceCount);
	for (int i = 0; i < m_FaceCount; i++)
	{
		const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		aabb.Grow(m_AABBs[i]);
	}

	for (int i = static_cast<int>(m_BVHPool.size()) - 1; i >= 0; i--)
	{
		auto &node = m_BVHPool[i];

		// Calculate new bounds of leaf nodes
		if (node.IsLeaf())
		{
			node.CalculateBounds(m_AABBs.data(), m_PrimitiveIndices.data());
		}
		else // Calculate new bounds of bvh nodes
		{
			const auto &leftNode = m_BVHPool[node.GetLeftFirst()];
			const auto &rightNode = m_BVHPool[node.GetLeftFirst() + 1];

			auto aabb = AABB();
			aabb.Grow(leftNode.bounds);
			aabb.Grow(rightNode.bounds);

			memcpy(node.bounds.bmin, aabb.bmin, 3 * sizeof(float));
			memcpy(node.bounds.bmax, aabb.bmax, 3 * sizeof(float));
		}
	}
}

void BVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	m_Vertices = vertices;
	m_Indices = indices;

	aabb = AABB();

	// Recalculate AABBs
	for (int i = 0; i < m_FaceCount; i++)
	{
		const uvec3 &idx = m_Indices[i];
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		aabb.Grow(m_AABBs[i]);
	}

	for (int i = static_cast<int>(m_BVHPool.size()) - 1; i >= 0; i--)
	{
		auto &node = m_BVHPool[i];
		// Calculate new bounds of leaf nodes
		if (node.IsLeaf())
		{
			node.CalculateBounds(m_AABBs.data(), m_PrimitiveIndices.data());
		}
		else // Calculate new bounds of bvh nodes
		{
			const auto &leftNode = m_BVHPool[node.GetLeftFirst()];
			const auto &rightNode = m_BVHPool[node.GetLeftFirst() + 1];

			auto aabb = AABB();
			aabb.Grow(leftNode.bounds);
			aabb.Grow(rightNode.bounds);
			memcpy(node.bounds.bmin, aabb.bmin, 3 * sizeof(float));
			memcpy(node.bounds.bmax, aabb.bmax, 3 * sizeof(float));
		}
	}
}

bool BVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx)
{
	if (m_Indices)
		return BVHNode::traverseBVH(origin, dir, t_min, t, primIdx, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices, m_Indices);
	return BVHNode::traverseBVH(origin, dir, t_min, t, primIdx, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices);
}
