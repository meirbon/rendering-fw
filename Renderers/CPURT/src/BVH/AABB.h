#pragma once

#include <immintrin.h>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <glm/simd/geometric.h>

class AABB
{
  public:
	AABB()
	{
		memset(&this->bmin4, 0, sizeof(glm::vec4));
		memset(&this->bmax4, 0, sizeof(glm::vec4));

		this->bmin[0] = 1e34f;
		this->bmin[1] = 1e34f;
		this->bmin[2] = 1e34f;

		this->bmax[0] = -1e34f;
		this->bmax[1] = -1e34f;
		this->bmax[2] = -1e34f;
	};

	AABB(__m128 mi, __m128 ma)
	{
		bmin4 = mi;
		bmax4 = ma;
		bmin[3] = bmax[3] = 0;
	}

	AABB(glm::vec3 mi, glm::vec3 ma)
	{
		bmin[0] = mi.x;
		bmin[1] = mi.y;
		bmin[2] = mi.z;
		bmin[3] = 0;

		bmax[0] = ma.x;
		bmax[1] = ma.y;
		bmax[2] = ma.z;
		bmax[3] = 0;
	}

	bool Intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max) const
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

		const __m128 t1 = _mm_mul_ps(_mm_sub_ps(bmin4, origin), dirInv);
		const __m128 t2 = _mm_mul_ps(_mm_sub_ps(bmax4, origin), dirInv);

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

	inline void Reset() { bmin4 = _mm_set_ps1(1e34f), bmax4 = _mm_set_ps1(-1e34f); }

	[[nodiscard]] inline bool Contains(const __m128 &p) const
	{
		union {
			__m128 va4;
			float va[4];
		};
		union {
			__m128 vb4;
			float vb[4];
		};
		va4 = _mm_sub_ps(p, bmin4), vb4 = _mm_sub_ps(bmax4, p);
		return ((va[0] >= 0) && (va[1] >= 0) && (va[2] >= 0) && (vb[0] >= 0) && (vb[1] >= 0) && (vb[2] >= 0));
	}

	inline void GrowSafe(const AABB &bb)
	{
		xMin = glm::min(xMin, bb.xMin);
		yMin = glm::min(yMin, bb.yMin);
		zMin = glm::min(zMin, bb.zMin);

		xMax = glm::max(xMax, bb.xMax);
		yMax = glm::max(yMax, bb.yMax);
		zMax = glm::max(zMax, bb.zMax);
	}

	inline void Grow(const AABB &bb)
	{
		bmin4 = _mm_min_ps(bmin4, bb.bmin4);
		bmax4 = _mm_max_ps(bmax4, bb.bmax4);
	}

	inline void Grow(const __m128 &p)
	{
		bmin4 = _mm_min_ps(bmin4, p);
		bmax4 = _mm_max_ps(bmax4, p);
	}

	inline void Grow(const __m128 min4, const __m128 max4)
	{
		bmin4 = _mm_min_ps(bmin4, min4);
		bmax4 = _mm_max_ps(bmax4, max4);
	}

	inline void Grow(const glm::vec3 &p)
	{
		__m128 p4 = _mm_setr_ps(p.x, p.y, p.z, 0);
		Grow(p4);
	}

	[[nodiscard]] AABB Union(const AABB &bb) const
	{
		AABB r;
		r.bmin4 = _mm_min_ps(bmin4, bb.bmin4);
		r.bmax4 = _mm_max_ps(bmax4, bb.bmax4);
		return r;
	}

	static AABB Union(const AABB &a, const AABB &b)
	{
		AABB r;
		r.bmin4 = _mm_min_ps(a.bmin4, b.bmin4);
		r.bmax4 = _mm_max_ps(a.bmax4, b.bmax4);
		return r;
	}

	[[nodiscard]] AABB Intersection(const AABB &bb) const
	{
		AABB r;
		r.bmin4 = _mm_max_ps(bmin4, bb.bmin4);
		r.bmax4 = _mm_min_ps(bmax4, bb.bmax4);
		return r;
	}

	[[nodiscard]] inline float Extend(const int axis) const { return bmax[axis] - bmin[axis]; }

	[[nodiscard]] inline float Minimum(const int axis) const { return bmin[axis]; }

	[[nodiscard]] inline float Maximum(const int axis) const { return bmax[axis]; }

	[[nodiscard]] inline float Volume() const
	{
		union {
			__m128 length4;
			float length[4];
		};
		length4 = _mm_sub_ps(this->bmax4, this->bmin4);
		return length[0] * length[1] * length[2];
	}

	[[nodiscard]] inline glm::vec3 Centroid() const
	{
		union {
			__m128 center;
			float c4[4];
		};
		center = Center();
		return glm::vec3(c4[0], c4[1], c4[2]);
	}

	[[nodiscard]] inline float Area() const
	{
		union {
			__m128 e4;
			float e[4];
		};
		e4 = _mm_sub_ps(bmax4, bmin4);
		return fmax(0.0f, e[0] * e[1] + e[0] * e[2] + e[1] * e[2]);
	}

	[[nodiscard]] inline glm::vec3 Lengths() const
	{
		union {
			__m128 length4;
			float length[4];
		};
		length4 = _mm_sub_ps(this->bmax4, this->bmin4);
		return glm::vec3(length[0], length[1], length[2]);
	}

	[[nodiscard]] inline int LongestAxis() const
	{
		int a = 0;
		if (Extend(1) > Extend(0))
			a = 1;
		if (Extend(2) > Extend(a))
			a = 2;
		return a;
	}

	union {
		struct
		{
			union {
				__m128 bmin4;
				float bmin[4];
				struct
				{
					float xMin, yMin, zMin;
					int leftFirst;
				};
			};
			union {
				__m128 bmax4;
				float bmax[4];
				struct
				{
					float xMax, yMax, zMax;
					int count;
				};
			};
		};
		__m128 bounds[2] = {_mm_set_ps(1e34f, 1e34f, 1e34f, 0), _mm_set_ps(-1e34f, -1e34f, -1e34f, 0)};
	};

	inline void SetBounds(const __m128 min4, const __m128 max4)
	{
		bmin4 = min4;
		bmax4 = max4;
	}

	[[nodiscard]] inline __m128 Center() const { return _mm_mul_ps(_mm_add_ps(bmin4, bmax4), _mm_set_ps1(0.5f)); }

	[[nodiscard]] inline float Center(unsigned int axis) const { return (bmin[axis] + bmax[axis]) * 0.5f; }
};