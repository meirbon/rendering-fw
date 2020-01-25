#pragma once

#include "AABB.h"

#include <atomic>
#include <utils/ArrayProxy.h>

namespace rfw
{
namespace bvh
{
struct MBVHTraversal
{
	int leftFirst;
	int count;
};

struct MBVHHit
{
	MBVHHit() { result = glm::bvec4(false, false, false, false); }

	union {
		glm::vec4 tminv;
		glm::ivec4 tminiv;

		__m128 tmin4;
		__m128i tmini4;

		int tmini[4];
		float tmin[4];
	};
	glm::bvec4 result;
};

class MBVHTree;
class MBVHNode
{
  public:
	MBVHNode()
	{
		for (int i = 0; i < 4; i++)
		{
			bminx[i] = bminy[i] = bminz[i] = 1e34f;
			bmaxx[i] = bmaxy[i] = bmaxz[i] = -1e34f;
		}

		childs = ivec4(-1);
		counts = ivec4(-1);
	}

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

	void set_bounds(unsigned int nodeIdx, const glm::vec3 &min, const glm::vec3 &max);

	void set_bounds(unsigned int nodeIdx, const AABB &bounds);

	MBVHHit intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float t) const;

	void merge_nodes(const BVHNode &node, const rfw::utils::ArrayProxy<BVHNode> bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr);

	void sort_results(const float *tmin, int &a, int &b, int &c, int &d) const;

	template <typename FUNC>
	static bool traverse_mbvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes, const uint *primIndices,
							  const FUNC &func)
	{
		bool valid = false;
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

			if (count > -1) // leaf node
			{
				for (int i = 0; i < count; i++)
				{
					const auto primID = primIndices[leftFirst + i];
					if (func(primID))
					{
						valid = true;
						*hit_idx = primID;
					}
				}
				continue;
			}

			const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, *t);
			for (int i = 3; i >= 0; i--) // reversed order, we want to check best nodes first
			{
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1 && nodes[leftFirst].childs[idx] >= 0)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
					todo[stackptr].count = nodes[leftFirst].counts[idx];
				}
			}
		}

		return valid;
	}

	template <typename FUNC>
	static bool traverse_mbvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float tmax, const MBVHNode *nodes, const uint *primIndices,
									 const FUNC &func)
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

			if (count > -1) // leaf node
			{
				for (int i = 0; i < count; i++)
				{
					const auto primID = primIndices[leftFirst + i];
					if (func(primID))
						return true;
				}
				continue;
			}

			const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, tmax);
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1 && nodes[leftFirst].childs[idx] >= 0)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
					todo[stackptr].count = nodes[leftFirst].counts[idx];
				}
			}
		}

		// Nothing occluding
		return false;
	}

	void validate(const rfw::utils::ArrayProxy<MBVHNode> nodes, const rfw::utils::ArrayProxy<uint> primIDs, uint maxPoolPtr, uint maxPrimIndex) const;

	void merge_node(const BVHNode &node, const rfw::utils::ArrayProxy<BVHNode> pool, int &numChildren);
};
} // namespace bvh
} // namespace rfw