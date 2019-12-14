#include "BVH/BVHNode.h"

#include <glm/glm.hpp>

#include "../Triangle.h"

#define MAX_PRIMS 3
#define MAX_DEPTH 64
#define BINS 11

using namespace glm;

BVHNode::BVHNode()
{
	SetLeftFirst(-1);
	SetCount(-1);
}

BVHNode::BVHNode(int leftFirst, int count, AABB bounds) : bounds(bounds)
{
	SetLeftFirst(leftFirst);
	SetCount(-1);
}

bool BVHNode::Intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max) const
{
#if 0
	const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;

	const glm::vec3 min = glm::min(t1, t2);
	const glm::vec3 max = glm::max(t1, t2);

	*t_min = glm::max(min.x, glm::max(min.y, min.z));
	*t_max = glm::min(max.x, glm::min(max.y, max.z));

	return *t_max >= 0.0f && *t_min < *t_max;
#else
	const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));

	const __m128 t1 = _mm_mul_ps(_mm_sub_ps(bounds.bmin4, origin), dirInv);
	const __m128 t2 = _mm_mul_ps(_mm_sub_ps(bounds.bmax4, origin), dirInv);

	union {
		__m128 tmin4;
		float tmin[4];
	};

	union {
		__m128 tmax4;
		float tmax[4];
	};

	tmin4 = _mm_min_ps(t1, t2);
	tmax4 = _mm_max_ps(t1, t2);

	*t_min = glm::max(tmin[0], glm::max(tmin[1], tmin[2]));
	*t_max = glm::min(tmax[0], glm::min(tmax[1], tmax[2]));

	return *t_max >= 0.0f && *t_min < *t_max;
#endif
}

void BVHNode::CalculateBounds(const AABB *aabbs, const unsigned int *primitiveIndices)
{
	AABB newBounds = {vec3(1e34f), vec3(-1e34f)};
	for (int idx = 0; idx < bounds.count; idx++)
		newBounds.Grow(aabbs[primitiveIndices[bounds.leftFirst + idx]]);

	bounds.xMin = newBounds.xMin - 1e-5f;
	bounds.yMin = newBounds.yMin - 1e-5f;
	bounds.zMin = newBounds.zMin - 1e-5f;
	bounds.xMax = newBounds.xMax + 1e-5f;
	bounds.yMax = newBounds.yMax + 1e-5f;
	bounds.zMax = newBounds.zMax + 1e-5f;
}

void BVHNode::Subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth, std::atomic_int &poolPtr)
{
	depth++;
	if (GetCount() < MAX_PRIMS || depth >= MAX_DEPTH)
		return; // this is a leaf node

	int left = -1;
	int right = -1;

	if (!Partition(aabbs, bvhTree, primIndices, left, right, poolPtr))
	{
		return;
	}

	this->bounds.leftFirst = left; // set pointer to children
	this->bounds.count = -1;	   // no primitives since we are no leaf node

	auto &leftNode = bvhTree[left];
	auto &rightNode = bvhTree[right];

	if (leftNode.bounds.count > 0)
	{
		leftNode.CalculateBounds(aabbs, primIndices);
		leftNode.Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
	}

	if (rightNode.bounds.count > 0)
	{
		rightNode.CalculateBounds(aabbs, primIndices);
		rightNode.Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
	}
}

void BVHNode::SubdivideMT(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::mutex *threadMutex, std::mutex *partitionMutex,
						  unsigned int *threadCount, unsigned int depth, std::atomic_int &poolPtr)
{
	depth++;
	if (GetCount() < MAX_PRIMS || depth >= MAX_DEPTH)
		return; // this is a leaf node

	int left = -1;
	int right = -1;

	if (!Partition(aabbs, bvhTree, primIndices, partitionMutex, left, right, poolPtr))
		return;

	this->bounds.leftFirst = left; // set pointer to children
	this->bounds.count = -1;	   // no primitives since we are no leaf node

	auto *leftNode = &bvhTree[left];
	auto *rightNode = &bvhTree[right];

	const auto subLeft = leftNode->GetCount() > 0;
	const auto subRight = rightNode->GetCount() > 0;

	if ((*threadCount) < std::thread::hardware_concurrency())
	{
		if (subLeft && subRight)
		{
			auto lock = std::lock_guard(*threadMutex);
			(*threadCount)++;
		}

		auto leftThread = std::async([&]() {
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->SubdivideMT(aabbs, bvhTree, primIndices, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		});

		rightNode->CalculateBounds(aabbs, primIndices);
		rightNode->SubdivideMT(aabbs, bvhTree, primIndices, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		leftThread.get();
	}
	else
	{
		if (subLeft)
		{
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->SubdivideMT(aabbs, bvhTree, primIndices, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		}

		if (subRight)
		{
			rightNode->CalculateBounds(aabbs, primIndices);
			rightNode->SubdivideMT(aabbs, bvhTree, primIndices, threadMutex, partitionMutex, threadCount, depth, poolPtr);
		}
	}
}

bool BVHNode::Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::mutex *partitionMutex, int &left, int &right,
						std::atomic_int &poolPtr)
{
	const int lFirst = bounds.leftFirst;
	int lCount = 0;
	int rFirst = bounds.leftFirst;
	int rCount = bounds.count;

	float parentNodeCost{}, lowestNodeCost = 1e34f, bestCoord{};
	int bestAxis{};

	parentNodeCost = bounds.Area() * static_cast<float>(bounds.count);
	const vec3 lengths = this->bounds.Lengths();
	for (int axis = 0; axis < 3; axis++)
	{
		for (int i = 1; i < BINS; i++)
		{
			const auto binOffset = static_cast<float>(i) / static_cast<float>(BINS);
			const float splitCoord = bounds.bmin[axis] + lengths[axis] * binOffset;
			int leftCount = 0, rightCount = 0;
			AABB leftBox = {vec3(1e34f), vec3(-1e34f)};
			AABB rightBox = {vec3(1e34f), vec3(-1e34f)};

			for (int idx = 0; idx < bounds.count; idx++)
			{
				const auto &aabb = aabbs[primIndices[lFirst + idx]];
				if (aabb.Centroid()[axis] <= splitCoord)
				{
					leftBox.Grow(aabb);
					leftCount++;
				}
				else
				{
					rightBox.Grow(aabb);
					rightCount++;
				}
			}

			const float leftArea = leftBox.Area();
			const float rightArea = rightBox.Area();

			const float splitNodeCost = leftArea * float(leftCount) + rightArea * float(rightCount);
			if (splitNodeCost < lowestNodeCost)
			{
				lowestNodeCost = splitNodeCost;
				bestCoord = splitCoord;
				bestAxis = axis;
			}
		}
	}

	if (parentNodeCost < lowestNodeCost)
		return false;

	for (int idx = 0; idx < bounds.count; idx++)
	{
		const auto &aabb = aabbs[primIndices[lFirst + idx]];

		if (aabb.Centroid()[bestAxis] <= bestCoord) // is on left side
		{
			std::swap(primIndices[lFirst + idx], primIndices[lFirst + lCount]);
			lCount++;
			rFirst++;
			rCount--;
		}
	}

	partitionMutex->lock();
	left = poolPtr++;
	right = poolPtr++;
	partitionMutex->unlock();

	bvhTree[left].bounds.leftFirst = lFirst;
	bvhTree[left].bounds.count = lCount;
	bvhTree[right].bounds.leftFirst = rFirst;
	bvhTree[right].bounds.count = rCount;

	return true;
}
bool BVHNode::Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int &left, int &right, std::atomic_int &poolPtr)
{
	const int lFirst = bounds.leftFirst;
	int lCount = 0;
	int rFirst = bounds.leftFirst;
	int rCount = bounds.count;

	float parentNodeCost{}, lowestNodeCost = 1e34f, bestCoord{};
	int bestAxis{};

	parentNodeCost = bounds.Area() * bounds.count;
	const vec3 lengths = this->bounds.Lengths();
	for (int i = 1; i < BINS; i++)
	{
		const auto binOffset = static_cast<float>(i) / static_cast<float>(BINS);
		for (int axis = 0; axis < 3; axis++)
		{
			const float splitCoord = bounds.bmin[axis] + lengths[axis] * binOffset;
			int leftCount = 0, rightCount = 0;
			AABB leftBox = {vec3(1e34f), vec3(-1e34f)};
			AABB rightBox = {vec3(1e34f), vec3(-1e34f)};

			for (int idx = 0; idx < bounds.count; idx++)
			{
				const auto &aabb = aabbs[primIndices[lFirst + idx]];
				if (aabb.Centroid()[axis] <= splitCoord)
				{
					leftBox.Grow(aabb);
					leftCount++;
				}
				else
				{
					rightBox.Grow(aabb);
					rightCount++;
				}
			}

			const float splitNodeCost = leftBox.Area() * static_cast<float>(leftCount) + rightBox.Area() * static_cast<float>(rightCount);
			if (splitNodeCost < lowestNodeCost)
			{
				lowestNodeCost = splitNodeCost;
				bestCoord = splitCoord;
				bestAxis = axis;
			}
		}
	}

	if (parentNodeCost < lowestNodeCost)
		return false;

	for (int idx = 0; idx < bounds.count; idx++)
	{
		const auto &aabb = aabbs[primIndices[lFirst + idx]];

		if (aabb.Centroid()[bestAxis] <= bestCoord) // is on left side
		{
			std::swap(primIndices[lFirst + idx], primIndices[lFirst + lCount]);
			lCount++;
			rFirst++;
			rCount--;
		}
	}

	left = static_cast<int>(poolPtr++);
	right = static_cast<int>(poolPtr++);

	bvhTree[left].bounds.leftFirst = lFirst;
	bvhTree[left].bounds.count = lCount;
	bvhTree[right].bounds.leftFirst = rFirst;
	bvhTree[right].bounds.count = rCount;

	return true;
}

void BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						  const unsigned int *primIndices, const glm::uvec3 *indices, const glm::vec3 *vertices)
{
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.GetCount() > -1)
		{
			for (int i = 0; i < node.GetCount(); i++)
			{
				const auto primIdx = primIndices[node.GetLeftFirst() + i];
				const auto idx = indices[primIdx];
				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z], 1e-5f))
					*hit_idx = primIdx;
			}
		}
		else
		{
			bool hitLeft = nodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
			bool hitRight = nodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

			if (hitLeft && hitRight)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst(), tNear1};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
			}
		}
	}
}

void BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						  const unsigned int *primIndices, const glm::vec3 *vertices)
{
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.GetCount() > -1)
		{
			for (int i = 0; i < node.GetCount(); i++)
			{
				const auto primIdx = primIndices[node.GetLeftFirst() + i];
				const auto idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);
				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					*hit_idx = primIdx;
			}
		}
		else
		{
			bool hitLeft = nodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
			bool hitRight = nodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

			if (hitLeft && hitRight)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst(), tNear1};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
			}
		}
	}
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								const glm::uvec3 *indices, const glm::vec3 *vertices)
{
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.GetCount() > -1)
		{
			for (int i = 0; i < node.GetCount(); i++)
			{
				const auto primIdx = primIndices[node.GetLeftFirst() + i];
				const auto idx = indices[primIdx];
				if (rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
		}
		else
		{
			bool hitLeft = nodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
			bool hitRight = nodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

			if (hitLeft && hitRight)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst(), tNear1};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
			}
		}
	}

	return false;
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								const glm::vec3 *vertices)
{
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.GetCount() > -1)
		{
			for (int i = 0; i < node.GetCount(); i++)
			{
				const auto primIdx = primIndices[node.GetLeftFirst() + i];
				const auto idx = uvec3(primIndices[primIdx] * 3) + uvec3(0, 1, 2);
				if (rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
		}
		else
		{
			bool hitLeft = nodes[node.GetLeftFirst()].Intersect(org, dirInverse, &tNear1, &tFar1);
			bool hitRight = nodes[node.GetLeftFirst() + 1].Intersect(org, dirInverse, &tNear2, &tFar2);

			if (hitLeft && hitRight)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst(), tNear1};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst(), tNear1};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1, tNear2};
			}
		}
	}

	return false;
}
