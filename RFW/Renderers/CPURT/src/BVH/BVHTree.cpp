#include "../PCH.h"

using namespace glm;
using namespace rfw;

#define EDGE_INTERSECTION 0

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount) : m_VertexCount(vertexCount), m_FaceCount(vertexCount / 3)
{
	m_Indices = nullptr;
	m_PoolPtr.store(0);
	reset();
	set_vertices(vertices);
}

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount) : m_VertexCount(vertexCount), m_FaceCount(faceCount)
{
	m_PoolPtr.store(0);
	reset();
	set_vertices(vertices, indices);
}

void BVHTree::construct_bvh(bool printBuildTime)
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
		rootNode.calculate_bounds(m_AABBs.data(), m_PrimitiveIndices.data());

		// rootNode.Subdivide(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), 1, m_PoolPtr);
		rootNode.subdivide_mt(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), m_BuildingThreads, 1, m_PoolPtr);

		if (m_PoolPtr > 2)
		{
			rootNode.bounds.count = -1;
			rootNode.set_left_first(2);
		}
		else
		{
			rootNode.bounds.count = static_cast<int>(m_FaceCount);
			rootNode.set_left_first(0);
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
	set_vertices(vertices);
	aabb = m_BVHPool[0].refit(m_BVHPool.data(), m_PrimitiveIndices.data(), m_AABBs.data());
}

void BVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	set_vertices(vertices, indices);
	aabb = m_BVHPool[0].refit(m_BVHPool.data(), m_PrimitiveIndices.data(), m_AABBs.data());
}

bool BVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx)
{
#if EDGE_INTERSECTION
	return BVHNode::traverse_bvh(origin, dir, t_min, t, primIdx, m_BVHPool.data(), m_PrimitiveIndices.data(), p0s.data(), edge1s.data(), edge2s.data());
#else
	if (m_Indices)
		return BVHNode::traverse_bvh(origin, dir, t_min, t, primIdx, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices, m_Indices);
	return BVHNode::traverse_bvh(origin, dir, t_min, t, primIdx, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices);
#endif
}

int BVHTree::traverse(cpurt::RayPacket4 &packet, float t_min, __m128 *hit_mask)
{
#if EDGE_INTERSECTION
	return BVHNode::traverse_bvh4(packet, t_min, m_BVHPool.data(), m_PrimitiveIndices.data(), p0s.data(), edge1s.data(), edge2s.data(), hit_mask);
#else
	if (m_Indices)
		return BVHNode::traverse_bvh4(packet, t_min, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices, m_Indices, hit_mask);
	return BVHNode::traverse_bvh4(packet, t_min, m_BVHPool.data(), m_PrimitiveIndices.data(), m_Vertices, hit_mask);
#endif
}

void BVHTree::set_vertices(const glm::vec4 *vertices)
{
	m_Vertices = vertices;

	aabb = AABB();

	// Recalculate data
	m_AABBs.resize(m_FaceCount);
	p0s.resize(m_FaceCount);
	edge1s.resize(m_FaceCount);
	edge2s.resize(m_FaceCount);

	rfw::utils::concurrency::parallel_for(0, m_FaceCount, [&](int i) {
		// for (int i = 0; i < m_FaceCount; i++)
		//{
		const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		aabb.grow(m_AABBs[i]);

		p0s[i] = vertices[idx.x];
		edge1s[i] = vertices[idx.y] - vertices[idx.x];
		edge2s[i] = vertices[idx.z] - vertices[idx.x];
	});
	// }

	for (int i = 0; i < 3; i++)
	{
		aabb.bmin[i] -= 1e-6f;
		aabb.bmax[i] += 1e-6f;
	}
}

void BVHTree::set_vertices(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	m_Vertices = vertices;
	m_Indices = indices;

	aabb = AABB();

	// Recalculate data
	m_AABBs.resize(m_FaceCount);
	p0s.resize(m_FaceCount);
	edge1s.resize(m_FaceCount);
	edge2s.resize(m_FaceCount);

	rfw::utils::concurrency::parallel_for(0, m_FaceCount, [&](int i) {
		const uvec3 &idx = m_Indices[i];
		m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		aabb.grow(m_AABBs[i]);

		p0s[i] = vertices[idx.x];
		edge1s[i] = vertices[idx.y] - vertices[idx.x];
		edge2s[i] = vertices[idx.z] - vertices[idx.x];
	});

	for (int i = 0; i < 3; i++)
	{
		aabb.bmin[i] -= 1e-5f;
		aabb.bmax[i] += 1e-5f;
	}
}
