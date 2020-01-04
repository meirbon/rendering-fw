#define GLM_FORCE_AVX
#include "BVH/MBVHTree.h"

#include "utils/Timer.h"

using namespace glm;
using namespace rfw;

MBVHTree::MBVHTree(BVHTree *orgTree)
{
	this->m_PrimitiveIndices = orgTree->m_PrimitiveIndices;
	this->m_OriginalTree = orgTree;
	this->aabb = orgTree->aabb;
}

void MBVHTree::constructBVH(bool printBuildTime)
{
	m_Tree.clear();
	// Worst case, this BVH becomes as big as the original
	m_Tree.resize(m_OriginalTree->m_BVHPool.size());
	if (m_OriginalTree->m_AABBs.empty())
		return;

	if (m_OriginalTree->m_PoolPtr <= 4) // Original tree first in single MBVH node
	{
		utils::Timer t{};
		m_PoolPtr.store(1);
		MBVHNode &mRootNode = m_Tree[0];

		for (int i = 0, s = static_cast<int>(m_OriginalTree->m_PoolPtr); i < s; i++)
		{
			BVHNode &curNode = m_OriginalTree->m_BVHPool[i];

			if (curNode.IsLeaf())
			{
				mRootNode.childs[i] = curNode.GetLeftFirst();
				mRootNode.counts[i] = curNode.GetCount();
				mRootNode.SetBounds(i, curNode.bounds);
			}
			else
			{
				mRootNode.childs[i] = 0;
				mRootNode.counts[i] = 0;
				const AABB invalidAABB = {glm::vec3(1e34f), glm::vec3(-1e34f)};
				mRootNode.SetBounds(i, invalidAABB);
			}
		}
		for (int i = m_OriginalTree->m_PoolPtr; i < 4; i++)
		{
			mRootNode.childs[i] = -1;
			mRootNode.counts[i] = 0;
		}

		if (printBuildTime)
			std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr.load() << std::endl;
	}
	else
	{
		utils::Timer t{};
		m_PoolPtr.store(1);
		MBVHNode &mRootNode = m_Tree[0];
		BVHNode &curNode = m_OriginalTree->m_BVHPool[0];
		m_BuildingThreads.store(1);
		mRootNode.MergeNodes(curNode, this->m_OriginalTree->m_BVHPool.data(), m_Tree.data(), m_PoolPtr);

		if (printBuildTime)
			std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_PoolPtr.load() << std::endl;
	}

#ifndef NDEBUG
	m_Tree[0].validate(m_Tree.data(), m_PrimitiveIndices.size(), m_PoolPtr);
#endif
}

void MBVHTree::refit(const glm::vec4 *vertices)
{
	m_OriginalTree->m_Vertices = vertices;

	// Recalculate AABBs
	m_OriginalTree->m_AABBs.resize(m_OriginalTree->m_FaceCount);
	for (int i = 0; i < m_OriginalTree->m_FaceCount; i++)
	{
		const glm::uvec3 idx = glm::uvec3(i * 3) + glm::uvec3(0, 1, 2);
		m_OriginalTree->m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	}

	// Calculate new bounds of leaf nodes
	for (auto &node : m_Tree)
	{
		for (int j = 0; j < 4; j++)
		{
			if (node.counts[j] <= 0)
				continue;

			auto aabb = AABB(vec3(1e34f), vec3(-1e34f));
			for (int k = node.childs[j], s = node.childs[j] + node.counts[j]; k < s; k++)
				aabb.Grow(m_OriginalTree->m_AABBs[m_OriginalTree->m_PrimitiveIndices[k]]);

			node.bminx[j] = aabb.bmin[0];
			node.bminy[j] = aabb.bmin[1];
			node.bminz[j] = aabb.bmin[2];

			node.bmaxx[j] = aabb.bmax[0];
			node.bmaxy[j] = aabb.bmax[1];
			node.bmaxz[j] = aabb.bmax[2];
		}
	}

	// Calculate new bounds of bvh nodes
	for (int i = static_cast<int>(m_Tree.size()) - 1; i >= 0; i--)
	{
		auto &node = m_Tree[i];
		for (int j = 0; j < 4; j++)
		{
			if (node.counts[j] >= 0 || node.childs[j] < i /* Child node cannot be at an earlier index */)
				continue;

			const auto &childNode = m_Tree[node.childs[j]];
			node.bminx[j] = min(childNode.bminx[0], min(childNode.bminx[1], childNode.bminx[2]));
			node.bminy[j] = min(childNode.bminy[0], min(childNode.bminy[1], childNode.bminy[2]));
			node.bminz[j] = min(childNode.bminz[0], min(childNode.bminz[1], childNode.bminz[2]));

			node.bmaxx[j] = max(childNode.bmaxx[0], max(childNode.bmaxx[1], childNode.bmaxx[2]));
			node.bmaxy[j] = max(childNode.bmaxy[0], max(childNode.bmaxy[1], childNode.bmaxy[2]));
			node.bmaxz[j] = max(childNode.bmaxz[0], max(childNode.bmaxz[1], childNode.bmaxz[2]));
		}
	}
}

void MBVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	m_OriginalTree->m_Vertices = vertices;
	m_OriginalTree->m_Indices = indices;

	// Recalculate AABBs
	for (int i = 0; i < m_OriginalTree->m_FaceCount; i++)
	{
		const glm::uvec3 &idx = m_OriginalTree->m_Indices[i];
		m_OriginalTree->m_AABBs[i] = triangle::getBounds(vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	}

	// Calculate new bounds of leaf nodes
	for (auto &node : m_Tree)
	{
		for (int j = 0; j < 4; j++)
		{
			if (node.counts[j] <= 0)
				continue;

			auto aabb = AABB(vec3(1e34f), vec3(-1e34f));
			for (int k = node.childs[j], s = node.childs[j] + node.counts[j]; k < s; k++)
				aabb.Grow(m_OriginalTree->m_AABBs[m_OriginalTree->m_PrimitiveIndices[k]]);

			node.bminx[j] = aabb.bmin[0];
			node.bminy[j] = aabb.bmin[1];
			node.bminz[j] = aabb.bmin[2];

			node.bmaxx[j] = aabb.bmax[0];
			node.bmaxy[j] = aabb.bmax[1];
			node.bmaxz[j] = aabb.bmax[2];
		}
	}

	// Calculate new bounds of bvh nodes
	for (int i = static_cast<int>(m_Tree.size()) - 1; i >= 0; i--)
	{
		auto &node = m_Tree[i];
		for (int j = 0; j < 4; j++)
		{
			if (node.counts[j] >= 0 || node.childs[j] < i /* Child node cannot be at an earlier index */)
				continue;

			const auto &childNode = m_Tree[node.childs[j]];
			node.bminx[j] = min(childNode.bminx[0], min(childNode.bminx[1], min(childNode.bminx[2], childNode.bminx[3])));
			node.bminy[j] = min(childNode.bminy[0], min(childNode.bminy[1], min(childNode.bminy[2], childNode.bminy[3])));
			node.bminz[j] = min(childNode.bminz[0], min(childNode.bminz[1], min(childNode.bminz[2], childNode.bminz[3])));

			node.bmaxx[j] = max(childNode.bmaxx[0], max(childNode.bmaxx[1], max(childNode.bmaxx[2], childNode.bmaxx[3])));
			node.bmaxy[j] = max(childNode.bmaxy[0], max(childNode.bmaxy[1], max(childNode.bmaxy[2], childNode.bmaxy[3])));
			node.bmaxz[j] = max(childNode.bmaxz[0], max(childNode.bmaxz[1], max(childNode.bmaxz[2], childNode.bmaxz[3])));
		}
	}
}

bool MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx)
{
	if (m_OriginalTree->m_Indices)
		return MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices,
									  m_OriginalTree->m_Indices);
	return MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices);
}
