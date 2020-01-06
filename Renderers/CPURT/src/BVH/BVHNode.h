#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

#include "BVH/AABB.h"
#include "../Ray.h"

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

	[[nodiscard]] bool is_leaf() const noexcept { return bounds.count >= 0; }

	bool intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, float min_t = 1e-6f) const;

	bool intersect(cpurt::RayPacket4 &packet4, __m128 *tmin_4, __m128 *tmax_4, float min_t = 1e-6f) const;

	bool intersect(cpurt::RayPacket8 &packet8, float min_t = 1e-6f) const;

	AABB refit(BVHNode *bvhTree, unsigned int *primIDs, AABB *aabbs);

	void set_count(int value) noexcept { bounds.count = value; }

	void set_left_first(unsigned int value) noexcept { bounds.leftFirst = value; }

	[[nodiscard]] inline int get_count() const noexcept { return bounds.count; }

	[[nodiscard]] inline int get_left_first() const noexcept { return bounds.leftFirst; }

	template <int BINS = 9, int MAX_DEPTH = 32, int MAX_PRIMITIVES = 3>
	void subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth, std::atomic_int &poolPtr)
	{
		depth++;
		if (get_count() < MAX_PRIMITIVES || depth >= MAX_DEPTH)
			return; // this is a leaf node

		auto left = -1;
		auto right = -1;

		if (!partition<BINS>(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
			return;

		this->bounds.leftFirst = left; // set pointer to children
		this->bounds.count = -1;	   // no primitives since we are no leaf node

		auto &left_node = bvhTree[left];
		auto &right_node = bvhTree[right];

		if (left_node.bounds.count > 0)
			left_node.subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);

		if (right_node.bounds.count > 0)
			right_node.subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
	}

	template <int BINS = 9, int MAX_DEPTH = 32, int MAX_PRIMITIVES = 3>
	void subdivide_mt(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::atomic_int &threadCount, unsigned int depth,
					  std::atomic_int &poolPtr)
	{
		depth++;
		if (get_count() < MAX_PRIMITIVES || depth >= MAX_DEPTH)
			return; // this is a leaf node

		int left = -1;
		int right = -1;

		if (!partition<BINS>(aabbs, bvhTree, primIndices, &left, &right, poolPtr))
			return;

		this->bounds.leftFirst = left; // set pointer to children
		this->bounds.count = -1;	   // no primitives since we are no leaf node

		auto *leftNode = &bvhTree[left];
		auto *rightNode = &bvhTree[right];

		const bool subLeft = leftNode->get_count() > 0;
		const bool subRight = rightNode->get_count() > 0;

		if (threadCount < std::thread::hardware_concurrency()) // Check if we need to create threads
		{
			if (subLeft && subRight)
			{
				threadCount.fetch_add(1);
				auto leftThread =
					std::async([&]() { leftNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr); });

				rightNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
				leftThread.get();
			}
			else if (subLeft)
				leftNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
			else if (subRight)
				rightNode->subdivide_mt<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, threadCount, depth, poolPtr);
		}
		else // No more need to create more threads
		{
			if (subLeft)
				leftNode->subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
			if (subRight)
				rightNode->subdivide<BINS, MAX_DEPTH, MAX_PRIMITIVES>(aabbs, bvhTree, primIndices, depth, poolPtr);
		}
	}

	template <int BINS> bool partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int *left, int *right, std::atomic_int &poolPtr)
	{
		const int lFirst = bounds.leftFirst;
		int lCount = 0;
		int rFirst = bounds.leftFirst;
		int rCount = bounds.count;

		float lowest_node_cost = 1e34f;
		float best_split = 0;
		int bestAxis = 0;

		auto best_left_box = AABB();
		auto best_right_box = AABB();

		float parent_node_cost = bounds.area() * static_cast<float>(bounds.count);
		const vec3 lengths = this->bounds.lengths();
		for (int axis = 0; axis < 3; axis++)
		{
			for (int i = 1; i < BINS; i++)
			{
				const auto bin_offset = static_cast<float>(i) / static_cast<float>(BINS);
				const auto split_offset = bounds.bmin[axis] + lengths[axis] * bin_offset;

				auto left_count = 0;
				auto right_count = 0;

				auto left_box = AABB();
				auto right_box = AABB();

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
		bvhTree[*left].bounds.leftFirst = lFirst;
		bvhTree[*left].bounds.count = lCount;

		bvhTree[*right].bounds.set_bounds(best_right_box);
		bvhTree[*right].bounds.leftFirst = rFirst;
		bvhTree[*right].bounds.count = rCount;

		return true;
	}

	void calculate_bounds(const AABB *aabbs, const unsigned int *primitiveIndices);

	template <typename FUNC>
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const FUNC &intersection)
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
				const bool hit_left = nodes[node.get_left_first()].intersect(org, dirInverse, &tNear1, &tFar1);
				const bool hit_right = nodes[node.get_left_first() + 1].intersect(org, dirInverse, &tNear2, &tFar2);

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

	template <typename FUNC>
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const FUNC &intersection)
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
				const bool hit_left = nodes[node.get_left_first()].intersect(org, dirInverse, &tNear1, &tFar1);
				const bool hit_right = nodes[node.get_left_first() + 1].intersect(org, dirInverse, &tNear2, &tFar2);

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

	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec3 *vertices, const glm::uvec3 *indices);
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec3 *vertices);
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices);
	static bool traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s);

	static int traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
							 const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask);
	static int traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned int *primIndices, const glm::vec4 *vertices,
							 __m128 *hit_mask);
	static int traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned int *primIndices, const glm::vec4 *vertices,
							 const glm::uvec3 *indices, __m128 *hit_mask);

	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const glm::vec3 *vertices, const glm::uvec3 *indices);
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const glm::vec3 *vertices);
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const glm::vec4 *vertices);
	static bool traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes,
									const unsigned int *primIndices, const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s);
};