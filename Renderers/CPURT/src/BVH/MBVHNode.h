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
		glm::vec4 bminx4;
		float bminx[4]{};
	};
	union {
		glm::vec4 bmaxx4;
		float bmaxx[4]{};
	};

	union {
		glm::vec4 bminy4;
		float bminy[4]{};
	};
	union {
		glm::vec4 bmaxy4;
		float bmaxy[4]{};
	};

	union {
		glm::vec4 bminz4;
		float bminz[4]{};
	};
	union {
		glm::vec4 bmaxz4;
		float bmaxz[4]{};
	};

	glm::ivec4 childs;
	glm::ivec4 counts;

	void SetBounds(unsigned int nodeIdx, const glm::vec3 &min, const glm::vec3 &max);

	void SetBounds(unsigned int nodeIdx, const AABB &bounds);

	inline MBVHHit intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t) const
	{
#if 1
		static const __m128i mask = _mm_set1_epi32(0xFFFFFFFCu);
		static const __m128i or_mask = _mm_set_epi32(0b11, 0b10, 0b01, 0b00);

		MBVHHit hit{};

		__m128 orgComponent = _mm_set1_ps(org.x);
		__m128 dirComponent = _mm_set1_ps(dirInverse.x);

		__m128 t1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bminx), orgComponent), dirComponent);
		__m128 t2 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bmaxx), orgComponent), dirComponent);

		hit.t_min = _mm_min_ps(t1, t2);
		__m128 t_max = _mm_max_ps(t1, t2);

		orgComponent = _mm_set1_ps(org.y);
		dirComponent = _mm_set1_ps(dirInverse.y);
		t1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bminy), orgComponent), dirComponent);
		t2 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bmaxy), orgComponent), dirComponent);

		hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
		t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

		orgComponent = _mm_set1_ps(org.z);
		dirComponent = _mm_set1_ps(dirInverse.z);
		t1 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bminz), orgComponent), dirComponent);
		t2 = _mm_mul_ps(_mm_sub_ps(_mm_load_ps(bmaxz), orgComponent), dirComponent);

		hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
		t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

		hit.t_mini = _mm_and_si128(hit.t_mini, mask);
		hit.t_mini = _mm_or_si128(hit.t_mini, or_mask);
		const __m128 greaterThan0 = _mm_cmpgt_ps(t_max, _mm_set1_ps(0.0f));
		const __m128 lessThanEqualMax = _mm_cmple_ps(hit.t_min, t_max);
		const __m128 lessThanT = _mm_cmplt_ps(hit.t_min, _mm_set1_ps(*t));

		const __m128 result = _mm_and_ps(greaterThan0, _mm_and_ps(lessThanEqualMax, lessThanT));
		const int resultMask = _mm_movemask_ps(result);
		hit.result = glm::bvec4(resultMask & 1, resultMask & 2, resultMask & 4, resultMask & 8);

		if (hit.tmin[0] > hit.tmin[1])
			std::swap(hit.tmin[0], hit.tmin[1]);
		if (hit.tmin[2] > hit.tmin[3])
			std::swap(hit.tmin[2], hit.tmin[3]);
		if (hit.tmin[0] > hit.tmin[2])
			std::swap(hit.tmin[0], hit.tmin[2]);
		if (hit.tmin[1] > hit.tmin[3])
			std::swap(hit.tmin[1], hit.tmin[3]);
		if (hit.tmin[2] > hit.tmin[3])
			std::swap(hit.tmin[2], hit.tmin[3]);

		return hit;
#else
		MBVHHit hit{};

		glm::vec4 t1 = (bminx4 - org.x) * dirInverse.x;
		glm::vec4 t2 = (bmaxx4 - org.x) * dirInverse.x;

		hit.tmin4 = glm::min(t1, t2);
		glm::vec4 tmax = glm::max(t1, t2);

		t1 = (bminy4 - org.y) * dirInverse.y;
		t2 = (bmaxy4 - org.y) * dirInverse.y;

		hit.tmin4 = glm::max(hit.tmin4, glm::min(t1, t2));
		tmax = glm::min(tmax, glm::max(t1, t2));

		t1 = (bminz4 - org.z) * dirInverse.z;
		t2 = (bmaxz4 - org.z) * dirInverse.z;

		hit.tmin4 = glm::max(hit.tmin4, glm::min(t1, t2));
		tmax = glm::min(tmax, glm::max(t1, t2));

		hit.tmini[0] = ((hit.tmini[0] & 0xFFFFFFFCu) | 0b00u);
		hit.tmini[1] = ((hit.tmini[1] & 0xFFFFFFFCu) | 0b01u);
		hit.tmini[2] = ((hit.tmini[2] & 0xFFFFFFFCu) | 0b10u);
		hit.tmini[3] = ((hit.tmini[3] & 0xFFFFFFFCu) | 0b11u);

		hit.result = greaterThan(tmax, glm::vec4(0.0f)) && lessThanEqual(hit.tmin4, tmax) && lessThan(hit.tmin4, glm::vec4(*t));

		if (hit.tmin[0] > hit.tmin[1])
			std::swap(hit.tmin[0], hit.tmin[1]);
		if (hit.tmin[2] > hit.tmin[3])
			std::swap(hit.tmin[2], hit.tmin[3]);
		if (hit.tmin[0] > hit.tmin[2])
			std::swap(hit.tmin[0], hit.tmin[2]);
		if (hit.tmin[1] > hit.tmin[3])
			std::swap(hit.tmin[1], hit.tmin[3]);
		if (hit.tmin[2] > hit.tmin[3])
			std::swap(hit.tmin[2], hit.tmin[3]);

		return hit;
#endif
	}

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

	void validate(MBVHNode* nodes, unsigned int maxPrimID, unsigned int maxPoolPtr);
};