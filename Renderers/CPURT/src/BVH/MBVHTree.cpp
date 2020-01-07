#include "../PCH.h"

using namespace glm;
using namespace rfw;

#define EDGE_INTERSECTION 1

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
		for (int i = 0, s = static_cast<int>(m_OriginalTree->m_PoolPtr); i < s; i++)
		{
			BVHNode &curNode = m_OriginalTree->m_BVHPool[i];

			if (curNode.is_leaf())
			{
				mRootNode.childs[i] = curNode.get_left_first();
				mRootNode.counts[i] = curNode.get_count();
				mRootNode.set_bounds(i, curNode.bounds);
			}
			else
			{
				mRootNode.childs[i] = 0;
				mRootNode.counts[i] = 0;
				const AABB invalidAABB = {glm::vec3(1e34f), glm::vec3(-1e34f)};
				mRootNode.set_bounds(i, invalidAABB);
			}
		}
		for (int i = m_OriginalTree->m_PoolPtr; i < 4; i++)
		{
			mRootNode.childs[i] = -1;
			mRootNode.counts[i] = 0;
		}

		m_Tree.resize(m_PoolPtr);

		if (printBuildTime)
			std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr.load() << std::endl;
	}
	else
	{
		BVHNode &curNode = m_OriginalTree->m_BVHPool[0];
		m_BuildingThreads.store(1);
		mRootNode.merge_nodes(curNode, this->m_OriginalTree->m_BVHPool.data(), m_Tree.data(), m_PoolPtr);

		if (printBuildTime)
			std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr.load() << std::endl;

		m_Tree.resize(m_PoolPtr);
	}

#ifndef NDEBUG
	m_Tree[0].validate(m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.size(), m_PoolPtr);
#endif
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
	return MBVHNode::traverse_mbvh(origin, dir, t_min, t, primIdx, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), m_OriginalTree->p0s.data(),
								  m_OriginalTree->edge1s.data(), m_OriginalTree->edge2s.data());
#else
	if (m_OriginalTree->m_Indices)
		return MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices,
									  m_OriginalTree->m_Indices);
	return MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices);
#endif
}

int MBVHTree::traverse(cpurt::RayPacket4 &packet, float t_min, __m128 *hit_mask)
{
	return MBVHNode::traverse_mbvh(packet, t_min, m_Tree.data(), m_OriginalTree->m_PrimitiveIndices.data(), m_OriginalTree->p0s.data(),
								  m_OriginalTree->edge1s.data(), m_OriginalTree->edge2s.data(), hit_mask);
}
