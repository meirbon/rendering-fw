#include "BVH/BVHNode.h"

#include <glm/glm.hpp>

#include "../Triangle.h"

#define MAX_PRIMS 3
#define MAX_DEPTH 48
#define BINS 9

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

bool BVHNode::Intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, const float min_t) const
{
	return bounds.Intersect(org, dirInverse, t_min, t_max, min_t);
}

bool BVHNode::intersect(cpurt::RayPacket4 &packet4, __m128 *tmin_4, __m128 *tmax_4, const float min_t) const
{
	return bounds.intersect(packet4, tmin_4, tmax_4, min_t);
}

bool BVHNode::intersect(cpurt::RayPacket8 &packet8, float min_t) const
{
	static const __m256 one8 = _mm256_set1_ps(1.0f);

	// const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m256 origin_x8 = _mm256_load_ps(packet8.origin_x);
	const __m256 origin_y8 = _mm256_load_ps(packet8.origin_y);
	const __m256 origin_z8 = _mm256_load_ps(packet8.origin_z);

	// const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m256 inv_direction_x8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_x));
	const __m256 inv_direction_y8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_y));
	const __m256 inv_direction_z8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_z));

	// const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const __m256 t1_8_x = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[0]), origin_x8), inv_direction_x8);
	const __m256 t1_8_y = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[1]), origin_y8), inv_direction_y8);
	const __m256 t1_8_z = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[2]), origin_z8), inv_direction_z8);

	// const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;
	const __m256 t2_8_x = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[0]), origin_x8), inv_direction_x8);
	const __m256 t2_8_y = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[1]), origin_y8), inv_direction_y8);
	const __m256 t2_8_z = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[2]), origin_z8), inv_direction_z8);

	// const glm::vec3 min = glm::min(t1, t2);
	const __m256 tmin_x8 = _mm256_min_ps(t1_8_x, t2_8_x);
	const __m256 tmin_y8 = _mm256_min_ps(t1_8_y, t2_8_y);
	const __m256 tmin_z8 = _mm256_min_ps(t1_8_z, t2_8_z);

	// const glm::vec3 max = glm::max(t1, t2);
	const __m256 tmax_x8 = _mm256_max_ps(t1_8_x, t2_8_x);
	const __m256 tmax_y8 = _mm256_max_ps(t1_8_y, t2_8_y);
	const __m256 tmax_z8 = _mm256_max_ps(t1_8_z, t2_8_z);

	//*t_min = glm::max(min.x, glm::max(min.y, min.z));
	const __m256 tmin_8 = _mm256_max_ps(tmin_x8, _mm256_max_ps(tmin_y8, tmin_z8));
	//*t_max = glm::min(max.x, glm::min(max.y, max.z));
	const __m256 tmax_8 = _mm256_min_ps(tmax_x8, _mm256_min_ps(tmax_y8, tmax_z8));

	const __m128 min_t4 = _mm_set1_ps(min_t);

	const __m128 tmin_4_0 = _mm256_extractf128_ps(tmin_8, 0);
	const __m128 tmin_4_1 = _mm256_extractf128_ps(tmin_8, 1);

	const __m128 tmax_4_0 = _mm256_extractf128_ps(tmax_8, 0);
	const __m128 tmax_4_1 = _mm256_extractf128_ps(tmax_8, 1);

	const __m128 mask4_0 = _mm_and_ps(_mm_cmpge_ps(tmax_4_0, min_t4), _mm_cmplt_ps(tmin_4_0, tmax_4_0));
	const __m128 mask4_1 = _mm_and_ps(_mm_cmpge_ps(tmax_4_1, min_t4), _mm_cmplt_ps(tmin_4_1, tmax_4_1));

	// return *t_max >= min_t && *t_min < *t_max;
	return _mm_movemask_ps(mask4_0) > 0 || _mm_movemask_ps(mask4_1) > 0;
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

	if (!Partition(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
		return;

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

void BVHNode::SubdivideMT(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::atomic_int &threadCount, unsigned int depth,
						  std::atomic_int &poolPtr)
{
	depth++;
	if (GetCount() < MAX_PRIMS || depth >= MAX_DEPTH)
		return; // this is a leaf node

	int left = -1;
	int right = -1;

	if (!Partition(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
		return;

	this->bounds.leftFirst = left; // set pointer to children
	this->bounds.count = -1;	   // no primitives since we are no leaf node

	auto *leftNode = &bvhTree[left];
	auto *rightNode = &bvhTree[right];

	const bool subLeft = leftNode->GetCount() > 0;
	const bool subRight = rightNode->GetCount() > 0;

	if (threadCount < std::thread::hardware_concurrency()) // Check if we need to create threads
	{
		if (subLeft && subRight)
		{
			threadCount.fetch_add(1);
			auto leftThread = std::async([&]() {
				leftNode->CalculateBounds(aabbs, primIndices);
				leftNode->SubdivideMT(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
			});

			rightNode->CalculateBounds(aabbs, primIndices);
			rightNode->SubdivideMT(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
			leftThread.get();
		}
		else if (subLeft)
		{
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->SubdivideMT(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
		}
		else if (subRight)
		{
			rightNode->CalculateBounds(aabbs, primIndices);
			rightNode->SubdivideMT(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
		}
	}
	else // No more need to create more threads
	{
		if (subLeft)
		{
			leftNode->CalculateBounds(aabbs, primIndices);
			leftNode->Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
		}

		if (subRight)
		{
			rightNode->CalculateBounds(aabbs, primIndices);
			rightNode->Subdivide(aabbs, bvhTree, primIndices, depth, poolPtr);
		}
	}
}

bool BVHNode::Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int *left, int *right, std::atomic_int &poolPtr)
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

	*left = poolPtr.fetch_add(2);
	*right = *left + 1;

	bvhTree[*left].bounds.leftFirst = lFirst;
	bvhTree[*left].bounds.count = lCount;
	bvhTree[*right].bounds.leftFirst = rFirst;
	bvhTree[*right].bounds.count = rCount;

	return true;
}

bool BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						  const unsigned int *primIndices, const glm::vec3 *vertices, const glm::uvec3 *indices)
{
	bool valid = false;
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
				{
					valid = true;
					*hit_idx = primIdx;
				}
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return valid;
}

bool BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						  const unsigned int *primIndices, const glm::vec3 *vertices)
{
	bool valid = false;
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
				{
					valid = true;
					*hit_idx = primIdx;
				}
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}
	return valid;
}

bool BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						  const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	bool valid = false;
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
				{
					valid = true;
					*hit_idx = primIdx;
				}
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}
	return valid;
}

bool BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						  const glm::vec4 *vertices)
{
	bool valid = false;
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
				{
					valid = true;
					*hit_idx = primIdx;
				}
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}
	return valid;
}

bool BVHNode::traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						  const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	bool valid = false;
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const auto dirInverse = 1.0f / dir;

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
				if (rfw::triangle::intersect_opt(org, dir, t_min, t, p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
				{
					valid = true;
					*hit_idx = primIdx;
				}
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}
	return valid;
}

int BVHNode::traverseBVH(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned *primIndices, const glm::vec3 *p0s,
						 const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask)
{
	bool valid = false;
	BVHTraversal todo[32];
	int stackPtr = 0;
	int hitMask = 0;
	__m128 tNear1, tFar1;
	__m128 tNear2, tFar2;

	__m128 store_mask = _mm_setzero_ps();

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
				const int mask = rfw::triangle::intersect4(packet, p0s[primIdx], edge1s[primIdx], edge2s[primIdx], &store_mask);
				if (mask != 0)
				{
					hitMask |= mask;
					*hit_mask = _mm_or_ps(*hit_mask, store_mask);
					_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
				}
			}
		}
		else
		{
			const bool hitLeft = nodes[node.GetLeftFirst()].intersect(packet, &tNear1, &tFar1, t_min);
			const bool hitRight = nodes[node.GetLeftFirst() + 1].intersect(packet, &tNear2, &tFar2, t_min);

			if (hitLeft && hitRight)
			{
				if (_mm_movemask_ps(_mm_cmplt_ps(tNear1, tNear2)) > 0 /* tNear1 < tNear2*/)
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return hitMask;
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								const glm::vec3 *vertices, const glm::uvec3 *indices)
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return false;
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								const glm::vec4 *vertices, const glm::uvec3 *indices)
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return false;
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								const glm::vec4 *vertices)
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return false;
}

bool BVHNode::traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const auto dirInverse = 1.0f / dir;

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
				if (rfw::triangle::intersect_opt(org, dir, t_min, &maxDist, p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
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
					todo[stackPtr] = {node.GetLeftFirst()};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst() + 1};
					stackPtr++;
					todo[stackPtr] = {node.GetLeftFirst()};
				}
			}
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.GetLeftFirst() + 1};
			}
		}
	}

	return false;
}
