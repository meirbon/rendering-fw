#pragma once

#include <glm/glm.hpp>
#include <immintrin.h>

#include "BVH/BVHNode.h"
#include "BVH/BVHTree.h"
#include "../Triangle.h"

struct MBVHTraversal
{
	int leftFirst;
	int count;
};

struct MBVHHit
{
	union {
		__m128 t_min;
		__m128i t_mini;
		glm::vec4 tmin4;
		float tmin[4];
		int tmini[4];
	};
	glm::bvec4 result;
};

class MBVHTree;
class MBVHNode
{
  public:
	MBVHNode() = default;

	~MBVHNode() = default;

	union {
		__m128 bminx_4;
		glm::vec4 bminx4;
		float bminx[4]{};
	};
	union {
		__m128 bmaxx_4;
		glm::vec4 bmaxx4;
		float bmaxx[4]{};
	};

	union {
		__m128 bminy_4;
		glm::vec4 bminy4;
		float bminy[4]{};
	};
	union {
		__m128 bmaxy_4;
		glm::vec4 bmaxy4;
		float bmaxy[4]{};
	};

	union {
		__m128 bminz_4;
		glm::vec4 bminz4;
		float bminz[4]{};
	};
	union {
		__m128 bmaxz_4;
		glm::vec4 bmaxz4;
		float bmaxz[4]{};
	};

	glm::ivec4 childs;
	glm::ivec4 counts;

	void SetBounds(unsigned int nodeIdx, const glm::vec3 &min, const glm::vec3 &max);

	void SetBounds(unsigned int nodeIdx, const AABB &bounds);

	MBVHHit intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t) const;

	void MergeNodes(const BVHNode &node, const BVHNode *bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr);

	void MergeNodesMT(const BVHNode &node, const BVHNode *bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr, std::atomic_int &threadCount,
					  bool thread = true);

	void GetBVHNodeInfo(const BVHNode &node, const BVHNode *pool, int &numChildren);

	void SortResults(const float *tmin, int &a, int &b, int &c, int &d) const;

	static bool traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static bool traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices);

	static bool traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								   const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static bool traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								   const unsigned int *primIndices, const glm::vec4 *vertices);

	void validate(MBVHNode *nodes, unsigned int maxPrimID, unsigned int maxPoolPtr);
};