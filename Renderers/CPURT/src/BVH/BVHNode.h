#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <future>

#include "BVH/AABB.h"

class BVHTree;

struct BVHTraversal
{
	int nodeIdx{};
	float tNear{};

	BVHTraversal(){};
	BVHTraversal(int nIdx, float t) : nodeIdx(nIdx), tNear(t) {}
};

struct BVHNode
{
  public:
	AABB bounds;

	glm::vec3 GetMin() const { return glm::make_vec3(bounds.bmin); }
	glm::vec3 GetMax() const { return glm::make_vec3(bounds.bmax); }

	BVHNode();

	BVHNode(int leftFirst, int count, AABB bounds);

	~BVHNode() = default;

	[[nodiscard]] inline bool IsLeaf() const noexcept { return bounds.count > -1; }

	bool Intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max) const;

	inline void SetCount(int value) noexcept { bounds.count = value; }

	inline void SetLeftFirst(unsigned int value) noexcept { bounds.leftFirst = value; }

	[[nodiscard]] inline int GetCount() const noexcept { return bounds.count; }

	[[nodiscard]] inline int GetLeftFirst() const noexcept { return bounds.leftFirst; }

	void Subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth, std::atomic_int &poolPtr);

	void SubdivideMT(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::mutex *threadMutex, std::mutex *partitionMutex,
					 unsigned int *threadCount, unsigned int depth, std::atomic_int &poolPtr);

	bool Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::mutex *partitionMutex, int &left, int &right, std::atomic_int &poolPtr);

	bool Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int &left, int &right, std::atomic_int &poolPtr);

	void CalculateBounds(const AABB *aabbs, const unsigned int *primitiveIndices);

	static void traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::uvec3 *indices, const glm::vec3 *vertices);

	static void traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec3 *vertices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::uvec3 *indices, const glm::vec3 *vertices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *vertices);
};