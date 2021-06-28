#pragma once

#include "aabb.h"

#include <atomic>
#include <thread>
#include <future>

namespace rfw
{
namespace bvh
{
class BVHTree;

struct BVHTraversal
{
	int nodeIdx{};

	BVHTraversal(){};
	BVHTraversal(int nIdx) : nodeIdx(nIdx) {}
};

struct BVHNode
{
  public:
	AABB bounds;

	[[nodiscard]] glm::vec3 get_min() const { return glm::make_vec3(bounds.bmin); }
	[[nodiscard]] glm::vec3 get_max() const { return glm::make_vec3(bounds.bmax); }

	BVHNode();

	BVHNode(int leftFirst, int count, AABB bounds);

	~BVHNode() = default;

	[[nodiscard]] bool is_leaf() const noexcept { return bounds.left_first >= 0; }

	bool intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, float min_t) const
	{
		return bounds.intersect(org, dirInverse, t_min, t_max, min_t);
	}

	AABB refit(BVHNode *bvhTree, uint *primIDs, AABB *aabbs);

	void set_count(int value) noexcept { bounds.count = value; }

	void set_left_first(unsigned int value) noexcept { bounds.left_first = value; }

	[[nodiscard]] inline int get_count() const noexcept { return bounds.count; }

	[[nodiscard]] inline int get_left_first() const noexcept { return bounds.left_first; }

	template <int BINS = 9, int MAX_DEPTH = 32, int MAX_PRIMITIVES = 3>
	void subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth,
				   std::atomic_int &poolPtr)
	{
		depth++;
		if (get_count() < MAX_PRIMITIVES || depth >= MAX_DEPTH)
			return; // this is a leaf node

		auto left = -1;
		auto right = -1;

		if (!partition<BINS>(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
			return;

		this->bounds.left_first = left; // set pointer to children
		this->bounds.count = -1;		// no primitives since we are no leaf node

		auto &left_node = bvhTree[left];
		auto &right_node = bvhTree[right];

		if (left_node.bounds.count > 0)
			left_node.subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);

		if (right_node.bounds.count > 0)
			right_node.subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
	}

	template <int BINS = 9, int MAX_DEPTH = 32, int MAX_PRIMITIVES = 3>
	void subdivide_mt(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::atomic_int &threadCount,
					  unsigned int depth, std::atomic_int &poolPtr)
	{
		depth++;
		if (get_count() < MAX_PRIMITIVES || depth >= MAX_DEPTH)
			return; // this is a leaf node

		int left = -1;
		int right = -1;

		if (!partition<BINS>(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
			return;

		this->bounds.left_first = left; // set pointer to children
		this->bounds.count = -1;		// no primitives since we are no leaf node

		auto *leftNode = &bvhTree[left];
		auto *rightNode = &bvhTree[right];

		const bool subLeft = leftNode->get_count() > 0;
		const bool subRight = rightNode->get_count() > 0;

		if (threadCount.load() < int(std::thread::hardware_concurrency())) // Check if we need to create threads
		{
			if (subLeft && subRight)
			{
				threadCount.fetch_add(1);
				auto leftThread = std::async(
					[&]() {
						leftNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices,
																				threadCount, depth, poolPtr);
					});

				rightNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount,
																		 depth, poolPtr);
				leftThread.get();
			}
			else if (subLeft)
				leftNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount, depth,
																		poolPtr);
			else if (subRight)
				rightNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount,
																		 depth, poolPtr);
		}
		else // No more need to create more threads
		{
			if (subLeft)
				leftNode->subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
			if (subRight)
				rightNode->subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
		}
	}

	template <int BINS>
	bool partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int *left, int *right,
				   std::atomic_int &poolPtr)
	{
		const int lFirst = bounds.left_first;
		int lCount = 0;
		int rFirst = bounds.left_first;
		int rCount = bounds.count;

		float lowest_node_cost = 1e34f;
		float best_split = 0;
		int bestAxis = 0;

		auto best_left_box = AABB();
		auto best_right_box = AABB();

		float parent_node_cost = bounds.area() * static_cast<float>(bounds.count);
		const vec3 lengths = this->bounds.lengths();

		const float bin_size = 1.0f / static_cast<float>(BINS + 2);

		for (int axis = 0; axis < 3; axis++)
		{
			for (int i = 1;
				 i < (BINS + 2 /* add 2 bins since we don't check walls of node and thus check 2 bins less */); i++)
			{
				const auto bin_offset = float(i) * bin_size;
				const auto split_offset = bounds.bmin[axis] + lengths[axis] * bin_offset;

				int left_count = 0;
				int right_count = 0;

				auto left_box = AABB::invalid();
				auto right_box = AABB::invalid();

				for (int idx = 0; idx < bounds.count; idx++)
				{
					const auto &aabb = aabbs[primIndices[lFirst + idx]];
					if (aabb.centroid()[axis] <= split_offset)
					{
						left_box.grow(aabb);
						left_count++;
					}
					else
					{
						right_box.grow(aabb);
						right_count++;
					}
				}

				const float leftArea = left_box.area();
				const float rightArea = right_box.area();

				const float splitNodeCost = leftArea * float(left_count) + rightArea * float(right_count);
				if (lowest_node_cost > splitNodeCost)
				{
					lowest_node_cost = splitNodeCost;
					best_split = split_offset;
					bestAxis = axis;

					best_left_box = left_box;
					best_right_box = right_box;
				}
			}
		}

		if (parent_node_cost < lowest_node_cost)
			return false;

		for (int idx = 0; idx < bounds.count; idx++)
		{
			const auto &aabb = aabbs[primIndices[lFirst + idx]];

			if (aabb.centroid()[bestAxis] <= best_split) // is on left side
			{
				std::swap(primIndices[lFirst + idx], primIndices[lFirst + lCount]);
				lCount++;
				rFirst++;
				rCount--;
			}
		}

		*left = poolPtr.fetch_add(2);
		*right = *left + 1;

		best_left_box.offset_by(1e-5f);
		best_right_box.offset_by(1e-5f);

		bvhTree[*left].bounds.set_bounds(best_left_box);
		bvhTree[*left].bounds.left_first = lFirst;
		bvhTree[*left].bounds.count = lCount;

		bvhTree[*right].bounds.set_bounds(best_right_box);
		bvhTree[*right].bounds.left_first = rFirst;
		bvhTree[*right].bounds.count = rCount;

		return true;
	}

	void calculate_bounds(const AABB *aabbs, const unsigned int *primitiveIndices);

	template <typename FUNC> // (int primIdx, __m128* store_mask) -> int
	static int traverse_bvh4(const float origin_x[4], const float origin_y[4], const float origin_z[4],
							 const float dir_x[4], const float dir_y[4], const float dir_z[4], float t[4],
							 int primID[4], const BVHNode *nodes, const unsigned int *primIndices, __m128 *hit_mask,
							 const FUNC &intersection)
	{
		using namespace simd;

		BVHTraversal todo[32];
		int stackPtr = 0;
		int hitMask = 0;
		simd::vector4 tNear1 = _mm_setzero_ps(), tFar1 = _mm_setzero_ps();
		simd::vector4 tNear2 = _mm_setzero_ps(), tFar2 = _mm_setzero_ps();

		const simd::vector4 inv_dir_x = ONE4 / vector4(dir_x);
		const simd::vector4 inv_dir_y = ONE4 / vector4(dir_y);
		const simd::vector4 inv_dir_z = ONE4 / vector4(dir_z);

		todo[stackPtr].nodeIdx = 0;
		while (stackPtr >= 0)
		{
			const auto &node = nodes[todo[stackPtr].nodeIdx];
			stackPtr--;

			if (node.get_count() > -1)
			{
				for (int i = 0; i < node.get_count(); i++)
				{
					const auto primIDx = primIndices[node.get_left_first() + i];
					__m128 store_mask = _mm_setzero_ps();
					int mask = intersection(primIDx, &store_mask);
					*hit_mask = _mm_or_ps(*hit_mask, store_mask);
					hitMask |= mask;
					_mm_maskstore_epi32(primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIDx));
				}
			}
			else
			{
				const int hitLeft = nodes[node.get_left_first()].bounds.intersect4(
					origin_x, origin_y, origin_z, reinterpret_cast<const float *>(&inv_dir_x),
					reinterpret_cast<const float *>(&inv_dir_y), reinterpret_cast<const float *>(&inv_dir_z), t,
					&tNear1, &tFar1);
				const int hitRight = nodes[node.get_left_first() + 1].bounds.intersect4(
					origin_x, origin_y, origin_z, reinterpret_cast<const float *>(&inv_dir_x),
					reinterpret_cast<const float *>(&inv_dir_y), reinterpret_cast<const float *>(&inv_dir_z), t,
					&tNear2, &tFar2);

				if (hitLeft > 0 && hitRight > 0)
				{
					if ((tNear1 < tNear2).move_mask() > 0 /* tNear1 < tNear2*/)
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
					}
					else
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
					}
				}
				else if (hitLeft)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
				else if (hitRight)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
			}
		}

		return hitMask;
	}

	template <typename FUNC> // (int primIdx) -> bool
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx,
							 const BVHNode *nodes, const unsigned int *primIndices, const FUNC &intersection)
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

			if (node.get_count() > -1)
			{
				for (int i = 0; i < node.get_count(); i++)
				{
					const auto primID = primIndices[node.get_left_first() + i];
					if (intersection(primID))
					{
						valid = true;
						*hit_idx = primID;
					}
				}
			}
			else
			{
				const bool hit_left = nodes[node.get_left_first()].intersect(org, dirInverse, &tNear1, &tFar1, *t);
				const bool hit_right = nodes[node.get_left_first() + 1].intersect(org, dirInverse, &tNear2, &tFar2, *t);

				if (hit_left && hit_right)
				{
					if (tNear1 < tNear2)
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
					}
					else
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
					}
				}
				else if (hit_left)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
				else if (hit_right)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
			}
		}

		return valid;
	}

	template <typename FUNC> // (int primIdx) -> bool
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist,
									const BVHNode *nodes, const unsigned int *primIndices, const FUNC &intersection)
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

			if (node.get_count() > -1)
			{
				for (int i = 0; i < node.get_count(); i++)
				{
					const auto primID = primIndices[node.get_left_first() + i];
					if (intersection(primID))
						return true;
				}
			}
			else
			{
				const bool hit_left = nodes[node.get_left_first()].intersect(org, dirInverse, &tNear1, &tFar1, maxDist);
				const bool hit_right =
					nodes[node.get_left_first() + 1].intersect(org, dirInverse, &tNear2, &tFar2, maxDist);

				if (hit_left && hit_right)
				{
					if (tNear1 < tNear2)
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
					}
					else
					{
						stackPtr++;
						todo[stackPtr] = {node.get_left_first() + 1};
						stackPtr++;
						todo[stackPtr] = {node.get_left_first()};
					}
				}
				else if (hit_left)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
				else if (hit_right)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
			}
		}

		return false;
	}
};
} // namespace bvh
} // namespace rfw