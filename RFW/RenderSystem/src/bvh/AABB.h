#pragma once

#include <MathIncludes.h>

namespace rfw
{
namespace bvh
{
class AABB
{
  public:
	AABB();

	AABB(simd::vector4 mi, simd::vector4 ma);

	AABB(glm::vec3 mi, glm::vec3 ma);
	static AABB invalid();

	bool intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, float min_t) const;
	int intersect4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
				   const float dir_y[4], const float dir_z[4], float t[4], simd::vector4 *t_min,
				   simd::vector4 *t_max) const;
	int intersect8(const float origin_x[8], const float origin_y[8], const float origin_z[8], const float dir_x[8],
				   const float dir_y[8], const float dir_z[8], float t[8], simd::vector8 *t_min,
				   simd::vector8 *t_max) const;

	void reset();
	[[nodiscard]] bool contains(const simd::vector4 &p) const;
	void grow_safe(const AABB &bb);
	void offset_by(const float offset);
	void offset_by(const float mi, const float ma);
	void grow(const AABB &bb);
	void grow(const simd::vector4 &p);
	void grow(const simd::vector4 min4, const simd::vector4 max4);

	void grow(const glm::vec3 &p);
	AABB union_of(const AABB &bb) const;
	static AABB union_of(const AABB &a, const AABB &b);
	[[nodiscard]] AABB intersection(const AABB &bb) const;
	[[nodiscard]] float extend(const int axis) const { return bmax[axis] - bmin[axis]; }
	[[nodiscard]] float minimum(const int axis) const { return bmin[axis]; }
	[[nodiscard]] float maximum(const int axis) const { return bmax[axis]; }
	[[nodiscard]] float volume() const;
	[[nodiscard]] glm::vec3 centroid() const;
	[[nodiscard]] float area() const;
	[[nodiscard]] glm::vec3 lengths() const;
	[[nodiscard]] int longest_axis() const;
	void set_bounds(const AABB &other);
	void set_bounds(const simd::vector4 min4, const simd::vector4 max4);
	[[nodiscard]] simd::vector4 center() const;
	[[nodiscard]] float center(unsigned int axis) const;

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
};
} // namespace bvh
} // namespace rfw
