#pragma once

namespace rfw
{
namespace bvh
{
class AABB
{
  public:
	AABB();

	AABB(__m128 mi, __m128 ma);

	AABB(glm::vec3 mi, glm::vec3 ma);

	bool intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, float min_t) const;
	void reset();
	[[nodiscard]] bool contains(const __m128 &p) const;
	void grow_safe(const AABB &bb);
	void offset_by(const float offset);
	void offset_by(const float mi, const float ma);
	void grow(const AABB &bb);
	void grow(const __m128 &p);
	void grow(const __m128 min4, const __m128 max4);
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

	void set_bounds(const AABB &other);

	void set_bounds(const __m128 min4, const __m128 max4);

	[[nodiscard]] __m128 center() const;

	[[nodiscard]] float center(unsigned int axis) const;
};
} // namespace bvh
} // namespace rfw
