#include <bvh/BVH.h>

namespace rfw
{
namespace bvh
{

AABB::AABB()
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = 1e34f;
		bmax[i] = -1e34f;
	}
};

AABB::AABB(simd::vector4 mi, simd::vector4 ma)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = mi[i];
		bmax[i] = ma[i];
	}
}

AABB::AABB(glm::vec3 mi, glm::vec3 ma)
{
	bmin[0] = mi.x;
	bmin[1] = mi.y;
	bmin[2] = mi.z;

	bmax[0] = ma.x;
	bmax[1] = ma.y;
	bmax[2] = ma.z;
}

AABB AABB::invalid() { return AABB(); }

bool AABB::intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *tmin, float *tmax, float t) const
{
#if 0 // Reference implementation
	const float tx1 = (bmin[0] - org.x) * dirInverse.x;
	const float tx2 = (bmax[0] - org.x) * dirInverse.x;

	*tmin = glm::min(tx1, tx2);
	*tmax = glm::max(tx1, tx2);

	const float ty1 = (bmin[1] - org.y) * dirInverse.y;
	const float ty2 = (bmax[1] - org.y) * dirInverse.y;

	*tmin = glm::max(*tmin, glm::min(ty1, ty2));
	*tmax = glm::min(*tmax, glm::max(ty1, ty2));

	const float tz1 = (bmin[2] - org.z) * dirInverse.z;
	const float tz2 = (bmax[2] - org.z) * dirInverse.z;

	*tmin = glm::max(*tmin, glm::min(tz1, tz2));
	*tmax = glm::min(*tmax, glm::max(tz1, tz2));
#else
	const simd::vector4 origin = simd::vector4(org.x, org.y, org.z, 0);
	const simd::vector4 dir_inv = simd::vector4(dirInverse.x, dirInverse.y, dirInverse.z, 0);

	const simd::vector4 mi = simd::vector4(bmin);
	const simd::vector4 ma = simd::vector4(bmax);

	const simd::vector4 t1 = (mi - origin) * dir_inv;
	const simd::vector4 t2 = (ma - origin) * dir_inv;

	const simd::vector4 tmin4 = min(t1, t2);
	const simd::vector4 tmax4 = max(t1, t2);
	*tmin = max(tmin4[0], max(tmin4[1], tmin4[2]));
	*tmax = min(tmax4[0], min(tmax4[1], tmax4[2]));
#endif

	// tmin must be less than tmax && t must be between tmin and tmax
	return (*tmax) > (*tmin) && (*tmin) < t;
}

int AABB::intersect4(const float origin_x[4], const float origin_y[4], const float origin_z[4],
					 const float inv_direction_x4[4], const float inv_direction_y4[4], const float inv_direction_z4[4],
					 float t[4], simd::vector4 *tmin, simd::vector4 *tmax) const
{
	using namespace simd;

	// const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	// const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));

	// const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const vector4 t1_4_x = (vector4(bmin[0]) - vector4(origin_x)) * inv_direction_x4;
	const vector4 t1_4_y = (vector4(bmin[1]) - vector4(origin_y)) * inv_direction_y4;
	const vector4 t1_4_z = (vector4(bmin[2]) - vector4(origin_z)) * inv_direction_z4;

	// const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;
	const vector4 t2_4_x = (vector4(bmax[0]) - vector4(origin_x)) * inv_direction_x4;
	const vector4 t2_4_y = (vector4(bmax[1]) - vector4(origin_y)) * inv_direction_y4;
	const vector4 t2_4_z = (vector4(bmax[2]) - vector4(origin_z)) * inv_direction_z4;

	// const glm::vec3 min = glm::min(t1, t2);
	const vector4 tmin_x4 = min(t1_4_x, t2_4_x);
	const vector4 tmin_y4 = min(t1_4_y, t2_4_y);
	const vector4 tmin_z4 = min(t1_4_z, t2_4_z);

	// const glm::vec3 max = glm::max(t1, t2);
	const vector4 tmax_x4 = max(t1_4_x, t2_4_x);
	const vector4 tmax_y4 = max(t1_4_y, t2_4_y);
	const vector4 tmax_z4 = max(t1_4_z, t2_4_z);

	//*t_min = glm::max(min.x, glm::max(min.y, min.z));
	*tmin = max(tmin_x4, max(tmin_y4, tmin_z4));
	//*t_max = glm::min(max.x, glm::min(max.y, max.z));
	*tmax = min(tmax_x4, min(tmax_y4, tmax_z4));

	// return (*tmax) > (*tmin) && (*tmin) < t;
	const vector4 mask = (*tmax > *tmin) & (*tmin < vector4(t));
	return mask.move_mask();
}

int AABB::intersect8(const float origin_x[8], const float origin_y[8], const float origin_z[8],
					 const float inv_direction_x8[8], const float inv_direction_y8[8], const float inv_direction_z8[8],
					 float t[8], simd::vector8 *t_min, simd::vector8 *t_max) const
{
	using namespace simd;

	// const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	// const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));

	const vector4 bmin0 = vector4(bmin[0]);
	const vector4 bmin1 = vector4(bmin[1]);
	const vector4 bmin2 = vector4(bmin[2]);

	const vector4 bmax0 = vector4(bmax[0]);
	const vector4 bmax1 = vector4(bmax[1]);
	const vector4 bmax2 = vector4(bmax[2]);

	// const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const vector8 t1_8_x = (vector8(bmin0, bmin0) - vector8(origin_x)) * inv_direction_x8;
	const vector8 t1_8_y = (vector8(bmin1, bmin1) - vector8(origin_y)) * inv_direction_y8;
	const vector8 t1_8_z = (vector8(bmin2, bmin2) - vector8(origin_z)) * inv_direction_z8;

	// const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;
	const vector8 t2_8_x = (vector8(bmax0, bmax0) - vector8(origin_x)) * inv_direction_x8;
	const vector8 t2_8_y = (vector8(bmax1, bmax1) - vector8(origin_y)) * inv_direction_y8;
	const vector8 t2_8_z = (vector8(bmax2, bmax2) - vector8(origin_z)) * inv_direction_z8;

	// const glm::vec3 min = glm::min(t1, t2);
	const vector8 tmin_x4 = min(t1_8_x, t2_8_x);
	const vector8 tmin_y4 = min(t1_8_y, t2_8_y);
	const vector8 tmin_z4 = min(t1_8_z, t2_8_z);

	// const glm::vec3 max = glm::max(t1, t2);
	const vector8 tmax_x4 = max(t1_8_x, t2_8_x);
	const vector8 tmax_y4 = max(t1_8_y, t2_8_y);
	const vector8 tmax_z4 = max(t1_8_z, t2_8_z);

	//*t_min = glm::max(min.x, glm::max(min.y, min.z));
	*t_min = max(tmin_x4, max(tmin_y4, tmin_z4));
	//*t_max = glm::min(max.x, glm::min(max.y, max.z));
	*t_max = min(tmax_x4, min(tmax_y4, tmax_z4));

	// return (*tmax) > (*tmin) && (*tmin) < t;
	const vector8 mask = (*t_max > *t_min) & (*t_min < vector8(t));
	return mask.move_mask();
}

void AABB::reset()
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = 1e34f;
		bmax[i] = -1e34f;
	}
}

bool AABB::contains(const simd::vector4 &p) const
{
	simd::vector4 va = p - simd::vector4(bmin);
	simd::vector4 vb = simd::vector4(bmax) - p;
	return ((va[0] >= 0) && (va[1] >= 0) && (va[2] >= 0) && (vb[0] >= 0) && (vb[1] >= 0) && (vb[2] >= 0));
}

void AABB::offset_by(const float offset)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] -= offset;
		bmax[i] += offset;
	}
}

void AABB::offset_by(const float mi, const float ma)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] += mi;
		bmax[i] += ma;
	}
}

void AABB::grow(const AABB &bb)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = glm::min(bmin[i], bb.bmin[i]);
		bmax[i] = glm::max(bmax[i], bb.bmax[i]);
	}
}

void AABB::grow(const simd::vector4 &p)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = glm::min(bmin[i], p[i]);
		bmax[i] = glm::max(bmax[i], p[i]);
	}
}

void AABB::grow(const simd::vector4 min4, const simd::vector4 max4)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = glm::min(bmin[i], min4[i]);
		bmax[i] = glm::max(bmax[i], max4[i]);
	}
}

void AABB::grow(const glm::vec3 &p) { grow(simd::vector4(p.x, p.y, p.z, 0)); }

AABB AABB::union_of(const AABB &bb) const
{
	AABB r;
	for (int i = 0; i < 3; i++)
	{
		r.bmin[i] = glm::min(bmin[i], bb.bmin[i]);
		r.bmax[i] = glm::max(bmax[i], bb.bmax[i]);
	}
	return r;
}

AABB AABB::union_of(const AABB &a, const AABB &b)
{
	AABB r;
	for (int i = 0; i < 3; i++)
	{
		r.bmin[i] = glm::min(a.bmin[i], b.bmin[i]);
		r.bmax[i] = glm::max(a.bmax[i], b.bmax[i]);
	}
	return r;
}

AABB AABB::intersection(const AABB &bb) const
{
	AABB r;
	for (int i = 0; i < 3; i++)
	{
		r.bmin[i] = glm::max(bmin[i], bb.bmin[i]);
		r.bmax[i] = glm::min(bmax[i], bb.bmax[i]);
	}
	return r;
}

float AABB::volume() const
{
	const simd::vector4 length = simd::vector4(bmax) - simd::vector4(bmin);
	return length[0] * length[1] * length[2];
}

glm::vec3 AABB::centroid() const
{
	simd::vector4 c = center();
	return glm::vec3(c[0], c[1], c[2]);
}

float AABB::area() const
{
	const simd::vector4 e = simd::vector4(bmax) - simd::vector4(bmin);
	return max(0.0f, e[0] * e[1] + e[0] * e[2] + e[1] * e[2]);
}

glm::vec3 AABB::lengths() const
{
	simd::vector4 length = simd::vector4(bmax) - simd::vector4(bmin);
	return glm::vec3(length[0], length[1], length[2]);
}

int AABB::longest_axis() const
{
	int a = 0;
	if (extend(1) > extend(0))
		a = 1;
	if (extend(2) > extend(a))
		a = 2;
	return a;
}

void AABB::set_bounds(const AABB &other)
{
	bmin[0] = other.bmin[0];
	bmin[1] = other.bmin[1];
	bmin[2] = other.bmin[2];

	bmax[0] = other.bmax[0];
	bmax[1] = other.bmax[1];
	bmax[2] = other.bmax[2];
}

void AABB::set_bounds(const simd::vector4 min4, const simd::vector4 max4)
{
	for (int i = 0; i < 3; i++)
	{
		bmin[i] = min4[i];
		bmax[i] = max4[i];
	}
}

simd::vector4 AABB::center() const { return (simd::vector4(bmin) + simd::vector4(bmax)) * 0.5f; }

float AABB::center(unsigned int axis) const { return (bmin[axis] + bmax[axis]) * 0.5f; }

} // namespace bvh
} // namespace rfw
