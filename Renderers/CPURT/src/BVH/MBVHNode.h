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
	}

	void MergeNodes(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree);

	void MergeNodesMT(const BVHNode &node, const BVHNode *bvhPool, MBVHTree *bvhTree, bool thread = true);

	void MergeNodes(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree);

	void MergeNodesMT(const BVHNode &node, const std::vector<BVHNode> &bvhPool, MBVHTree *bvhTree, bool thread = true);

	void GetBVHNodeInfo(const BVHNode &node, const BVHNode *pool, int &numChildren);

	void SortResults(const float *tmin, int &a, int &b, int &c, int &d) const;

	static void traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static void traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices);

	static bool traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								   const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices);
	static bool traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
								   const unsigned int *primIndices, const glm::vec4 *vertices);
};