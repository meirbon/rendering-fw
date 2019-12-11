#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"

#include <thread>
#include <future>

using namespace glm;
using namespace rfw;

void MBVHNode::SetBounds(unsigned int nodeIdx, const vec3 &min, const vec3 &max)
{
	this->bminx[nodeIdx] = min.x;
	this->bminy[nodeIdx] = min.y;
	this->bminz[nodeIdx] = min.z;

	this->bmaxx[nodeIdx] = max.x;
	this->bmaxy[nodeIdx] = max.y;
	this->bmaxz[nodeIdx] = max.z;
}

void MBVHNode::SetBounds(unsigned int nodeIdx, const AABB &bounds)
{
	this->bminx[nodeIdx] = bounds.xMin;
	this->bminy[nodeIdx] = bounds.yMin;
	this->bminz[nodeIdx] = bounds.zMin;

	this->bmaxx[nodeIdx] = bounds.xMax;
	this->bmaxy[nodeIdx] = bounds.yMax;
	this->bmaxz[nodeIdx] = bounds.zMax;
}

void MBVHNode::MergeNodes(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->counts[idx] == -1)
		{ // not a leaf
			const BVHNode &curNode = bvhPool[this->childs[idx]];
			if (curNode.IsLeaf())
			{
				this->counts[idx] = curNode.GetCount();
				this->childs[idx] = curNode.GetLeftFirst();
				this->SetBounds(idx, curNode.bounds);
			}
			else
			{
				const uint newIdx = bvhTree->m_FinalPtr++;
				MBVHNode &newNode = bvhTree->m_Tree[newIdx];
				this->childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
				this->counts[idx] = -1;
				this->SetBounds(idx, curNode.bounds);
				newNode.MergeNodes(curNode, bvhPool, bvhTree);
			}
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->counts[idx] = 0;
	}
}

void MBVHNode::MergeNodesMT(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree, bool thread)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	int threadCount = 0;
	std::vector<std::future<void>> threads;

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->counts[idx] = 0;
	}

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->counts[idx] == -1)
		{ // not a leaf
			const BVHNode *curNode = &bvhPool[this->childs[idx]];

			if (curNode->IsLeaf())
			{
				this->counts[idx] = curNode->GetCount();
				this->childs[idx] = curNode->GetLeftFirst();
				this->SetBounds(idx, curNode->bounds);
				continue;
			}

			bvhTree->m_PoolPtrMutex.lock();
			const auto newIdx = bvhTree->m_FinalPtr++;
			bvhTree->m_PoolPtrMutex.unlock();

			MBVHNode *newNode = &bvhTree->m_Tree[newIdx];
			this->childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
			this->counts[idx] = -1;
			this->SetBounds(idx, curNode->bounds);

			if (bvhTree->m_ThreadLimitReached || !thread)
			{
				newNode->MergeNodesMT(*curNode, bvhPool, bvhTree, !thread);
			}
			else
			{
				bvhTree->m_ThreadMutex.lock();
				bvhTree->m_BuildingThreads++;
				if (bvhTree->m_BuildingThreads > std::thread::hardware_concurrency())
					bvhTree->m_ThreadLimitReached = true;
				bvhTree->m_ThreadMutex.unlock();

				threadCount++;
				threads.push_back(std::async([newNode, curNode, bvhPool, bvhTree]() { newNode->MergeNodesMT(*curNode, bvhPool, bvhTree); }));
			}
		}
	}

	for (int i = 0; i < threadCount; i++)
	{
		threads[i].get();
	}
}

void MBVHNode::MergeNodes(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool.data(), numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->counts[idx] == -1)
		{ // not a leaf
			const BVHNode &curNode = bvhPool[this->childs[idx]];
			if (curNode.IsLeaf())
			{
				this->counts[idx] = curNode.GetCount();
				this->childs[idx] = curNode.GetLeftFirst();
				this->SetBounds(idx, curNode.bounds);
			}
			else
			{
				const uint newIdx = bvhTree->m_FinalPtr++;
				MBVHNode &newNode = bvhTree->m_Tree[newIdx];
				this->childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
				this->counts[idx] = -1;
				this->SetBounds(idx, curNode.bounds);
				newNode.MergeNodes(curNode, bvhPool, bvhTree);
			}
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->counts[idx] = 0;
	}
}

void MBVHNode::MergeNodesMT(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree, bool thread)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool.data(), numChildren);

	int threadCount = 0;
	std::vector<std::future<void>> threads{};

	// Invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->counts[idx] = 0;
	}

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->counts[idx] == -1)
		{ // not a leaf
			const BVHNode *curNode = &bvhPool[this->childs[idx]];

			if (curNode->IsLeaf())
			{
				this->counts[idx] = curNode->GetCount();
				this->childs[idx] = curNode->GetLeftFirst();
				this->SetBounds(idx, curNode->bounds);
				continue;
			}

			bvhTree->m_PoolPtrMutex.lock();
			const auto newIdx = bvhTree->m_FinalPtr++;
			bvhTree->m_PoolPtrMutex.unlock();

			MBVHNode *newNode = &bvhTree->m_Tree[newIdx];
			this->childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
			this->counts[idx] = -1;
			this->SetBounds(idx, curNode->bounds);

			if (bvhTree->m_ThreadLimitReached || !thread)
			{
				newNode->MergeNodesMT(*curNode, bvhPool, bvhTree, !thread);
			}
			else
			{
				bvhTree->m_ThreadMutex.lock();
				bvhTree->m_BuildingThreads++;
				if (bvhTree->m_BuildingThreads > std::thread::hardware_concurrency())
					bvhTree->m_ThreadLimitReached = true;
				bvhTree->m_ThreadMutex.unlock();

				threadCount++;
				threads.push_back(std::async([newNode, curNode, bvhPool, bvhTree]() { newNode->MergeNodesMT(*curNode, bvhPool, bvhTree); }));
			}
		}
	}

	for (int i = 0; i < threadCount; i++)
	{
		threads[i].get();
	}
}

void MBVHNode::GetBVHNodeInfo(const BVHNode &node, const BVHNode *pool, int &numChildren)
{
	// Starting values
	childs = ivec4(-1);
	counts = ivec4(-1);
	numChildren = 0;

	if (node.IsLeaf())
	{
		std::cout << "This node shouldn't be a leaf."
				  << "MBVHNode" << std::endl;
		return;
	}

	const BVHNode &leftNode = pool[node.GetLeftFirst()];
	const BVHNode &rightNode = pool[node.GetLeftFirst() + 1];

	if (leftNode.IsLeaf())
	{
		// node only has a single child
		const int idx = numChildren++;
		SetBounds(idx, leftNode.bounds);
		childs[idx] = leftNode.GetLeftFirst();
		counts[idx] = leftNode.GetCount();
	}
	else
	{
		// Node has 2 children
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;
		childs[idx1] = leftNode.GetLeftFirst();
		childs[idx2] = leftNode.GetLeftFirst() + 1;
	}

	if (rightNode.IsLeaf())
	{
		// Node only has a single child
		const int idx = numChildren++;
		SetBounds(idx, rightNode.bounds);
		childs[idx] = rightNode.GetLeftFirst();
		counts[idx] = rightNode.GetCount();
	}
	else
	{
		// Node has 2 children
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;
		SetBounds(idx1, pool[rightNode.GetLeftFirst()].bounds);
		SetBounds(idx2, pool[rightNode.GetLeftFirst() + 1].bounds);
		childs[idx1] = rightNode.GetLeftFirst();
		childs[idx2] = rightNode.GetLeftFirst() + 1;
	}
}

void MBVHNode::SortResults(const float *tmin, int &a, int &b, int &c, int &d) const
{
	if (tmin[a] > tmin[b])
		std::swap(a, b);
	if (tmin[c] > tmin[d])
		std::swap(c, d);
	if (tmin[a] > tmin[c])
		std::swap(a, c);
	if (tmin[b] > tmin[d])
		std::swap(b, d);
	if (tmin[b] > tmin[c])
		std::swap(b, c);
}
void MBVHNode::traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count > -1)
		{ // leaf node
			for (int i = 0; i < count; i++)
			{
				const glm::uint primIdx = primIndices[leftFirst + i];
				const glm::uvec3 &idx = indices[primIdx];

				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					*hit_idx = primIdx;
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}
}
void MBVHNode::traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							const unsigned int *primIndices, const glm::vec4 *vertices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count > -1)
		{ // leaf node
			for (int i = 0; i < count; i++)
			{
				const glm::uint primIdx = primIndices[leftFirst + i];
				const glm::uvec3 idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);

				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					*hit_idx = primIdx;
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}
}

bool MBVHNode::traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								  const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		struct MBVHTraversal mTodo = todo[stackptr];
		stackptr--;

		if (mTodo.count > -1)
		{ // leaf node
			for (int i = 0; i < mTodo.count; i++)
			{
				const int primIdx = primIndices[mTodo.leftFirst + i];
				const uvec3 &idx = indices[primIdx];

				if (triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
			continue;
		}

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist);
		if (hit.result[0] || hit.result[1] || hit.result[2] || hit.result[3])
		{
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[mTodo.leftFirst].childs[idx];
					todo[stackptr].count = nodes[mTodo.leftFirst].counts[idx];
				}
			}
		}
	}

	// Nothing occluding
	return false;
}

bool MBVHNode::traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								  const unsigned int *primIndices, const glm::vec4 *vertices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		struct MBVHTraversal mTodo = todo[stackptr];
		stackptr--;

		if (mTodo.count > -1)
		{ // leaf node
			for (int i = 0; i < mTodo.count; i++)
			{
				const int primIdx = primIndices[mTodo.leftFirst + i];
				const uvec3 idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);

				if (triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
			continue;
		}

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist);
		if (hit.result[0] || hit.result[1] || hit.result[2] || hit.result[3])
		{
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[mTodo.leftFirst].childs[idx];
					todo[stackptr].count = nodes[mTodo.leftFirst].counts[idx];
				}
			}
		}
	}

	// Nothing occluding
	return false;
}
