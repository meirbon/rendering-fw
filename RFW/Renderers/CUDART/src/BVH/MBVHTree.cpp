#include "../PCH.h"

using namespace glm;
using namespace rfw;

namespace rfw::bvh
{

#define EDGE_INTERSECTION 0

MBVHTree::MBVHTree(BVHTree *orgTree) { this->m_OriginalTree = orgTree; }

void MBVHTree::construct_bvh(bool printBuildTime)
{
	m_Tree.clear();
	// Worst case, this BVH becomes as big as the original
	m_Tree.resize(m_OriginalTree->m_BVHPool.size());
	if (m_OriginalTree->m_AABBs.empty())
		return;

	utils::Timer t{};
	m_PoolPtr.store(1);
	MBVHNode &mRootNode = m_Tree[0];

	if (m_OriginalTree->m_PoolPtr <= 4) // Original tree first in single MBVH node
	{
		int num_children = 0;
		mRootNode.merge_node(m_OriginalTree->m_BVHPool[0], m_OriginalTree->m_BVHPool, num_children);
	}
	else
	{
		m_BuildingThreads.store(1);
		mRootNode.merge_nodes(m_OriginalTree->m_BVHPool[0], m_OriginalTree->m_BVHPool, m_Tree.data(), m_PoolPtr);
	}

	m_Tree.resize(m_PoolPtr);
	if (printBuildTime)
		std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr.load() << std::endl;

	if (m_OriginalTree->m_Indices != nullptr)
		m_Tree[0].validate(m_Tree, m_OriginalTree->m_PrimitiveIndices, m_PoolPtr, m_OriginalTree->m_FaceCount);
	else
		m_Tree[0].validate(m_Tree, m_OriginalTree->m_PrimitiveIndices, m_PoolPtr, m_OriginalTree->m_VertexCount / 3);
}

void MBVHTree::refit(const glm::vec4 *vertices)
{
	m_OriginalTree->refit(vertices);
	construct_bvh();
}

void MBVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	m_OriginalTree->refit(vertices, indices);
	construct_bvh();
}

bool MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx)
{
#if EDGE_INTERSECTION
	return MBVHNode::traverse_mbvh(origin, dir, t_min, t, primIdx, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
		return triangle::intersect_opt(origin, dir, t_min, t, m_OriginalTree->p0s[primID], m_OriginalTree->edge1s[primID], m_OriginalTree->edge2s[primID]);
	});
#else
	if (m_OriginalTree->m_Indices)
	{
		return MBVHNode::traverse_mbvh(origin, dir, t_min, t, primIdx, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
			const uvec3 &idx = m_OriginalTree->m_Indices[primID];

			return triangle::intersect(origin, dir, t_min, t, m_OriginalTree->m_Vertices[idx.x], m_OriginalTree->m_Vertices[idx.y],
									   m_OriginalTree->m_Vertices[idx.z]);
		});
	}
	else
	{
		return MBVHNode::traverse_mbvh(origin, dir, t_min, t, primIdx, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
			const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
			return triangle::intersect(origin, dir, t_min, t, m_OriginalTree->m_Vertices[idx.x], m_OriginalTree->m_Vertices[idx.y],
									   m_OriginalTree->m_Vertices[idx.z]);
		});
	}
#endif
}

bool MBVHTree::traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float tmax)
{
#if EDGE_INTERSECTION
	return MBVHNode::traverse_mbvh(origin, dir, t_min, t, primIdx, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
		return triangle::intersect_opt(origin, dir, t_min, t, m_OriginalTree->p0s[primID], m_OriginalTree->edge1s[primID], m_OriginalTree->edge2s[primID]);
	});
#else
	if (m_OriginalTree->m_Indices)
	{
		return MBVHNode::traverse_mbvh_shadow(origin, dir, t_min, tmax, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
			const auto idx = m_OriginalTree->m_Indices[primID];
			return triangle::intersect_opt(origin, dir, t_min, &tmax, m_OriginalTree->m_Vertices[idx.x], m_OriginalTree->m_Vertices[idx.y],
										   m_OriginalTree->m_Vertices[idx.z]);
		});
	}
	else
	{
		return MBVHNode::traverse_mbvh_shadow(origin, dir, t_min, tmax, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), [&](uint primID) {
			const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
			return triangle::intersect_opt(origin, dir, t_min, &tmax, m_OriginalTree->m_Vertices[idx.x], m_OriginalTree->m_Vertices[idx.y],
										   m_OriginalTree->m_Vertices[idx.z]);
		});
	}
#endif
}

} // namespace rfw::bvh