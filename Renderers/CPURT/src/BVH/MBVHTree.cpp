#include "BVH/MBVHTree.h"

#define PRINT_BUILD_TIME 1

#include "utils/Timer.h"

using namespace rfw;

MBVHTree::MBVHTree(BVHTree *orgTree)
{
	this->m_PrimitiveIndices = orgTree->m_PrimitiveIndices;
	this->m_OriginalTree = orgTree;
	this->aabb = orgTree->aabb;
}

void MBVHTree::constructBVH()
{
	m_Tree.clear();
	m_Tree.resize(m_OriginalTree->m_FaceCount * 2);
	if (!m_OriginalTree->m_AABBs.empty())
	{
		utils::Timer t{};
		m_FinalPtr = 1;
		MBVHNode &mRootNode = m_Tree[0];
		BVHNode &curNode = m_OriginalTree->m_BVHPool[0];
		mRootNode.MergeNodes(curNode, this->m_OriginalTree->m_BVHPool, this);

		std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << m_FinalPtr << std::endl;
	}
}

void MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *t, int *primIdx)
{
	if (m_OriginalTree->m_Indices)
		MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices, m_OriginalTree->m_Indices);
	else
		MBVHNode::traverseMBVH(origin, dir, t_min, t, primIdx, m_Tree.data(), m_PrimitiveIndices.data(), m_OriginalTree->m_Vertices);
}
