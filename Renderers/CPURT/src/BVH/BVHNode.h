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

	glm::vec3 GetMin() const { return glm::make_vec3(bounds.bmin); }
	glm::vec3 GetMax() const { return glm::make_vec3(bounds.bmax); }

	BVHNode();

	BVHNode(int leftFirst, int count, AABB bounds);

	~BVHNode() = default;

	[[nodiscard]] inline bool IsLeaf() const noexcept { return bounds.count >= 0; }

	bool Intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, float min_t = 1e-6f) const;

	bool intersect(cpurt::RayPacket4 &packet4, __m128 *tmin_4, __m128 *tmax_4, float min_t = 1e-6f) const;

	bool intersect(cpurt::RayPacket8 &packet8, float min_t = 1e-6f) const;

	inline void SetCount(int value) noexcept { bounds.count = value; }

	inline void SetLeftFirst(unsigned int value) noexcept { bounds.leftFirst = value; }

	[[nodiscard]] inline int GetCount() const noexcept { return bounds.count; }

	[[nodiscard]] inline int GetLeftFirst() const noexcept { return bounds.leftFirst; }

	void Subdivide(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, unsigned int depth, std::atomic_int &poolPtr);

	void SubdivideMT(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, std::atomic_int &threadCount, unsigned int depth,
					 std::atomic_int &poolPtr);

	bool Partition(const AABB *aabbs, BVHNode *bvhTree, unsigned int *primIndices, int *left, int *right, std::atomic_int &poolPtr);

	void CalculateBounds(const AABB *aabbs, const unsigned int *primitiveIndices);

	static bool traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec3 *vertices, const glm::uvec3 *indices);

	static bool traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec3 *vertices);

	static bool traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);

	static bool traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec4 *vertices);

	static bool traverseBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
							const unsigned int *primIndices, const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s);

	static int traverseBVH(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
						   const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *vertices, const glm::uvec3 *indices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *vertices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec4 *vertices, const glm::uvec3 *indices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec4 *vertices);

	static bool traverseBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s);
};