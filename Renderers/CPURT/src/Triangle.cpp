#define GLM_FORCE_AVX

#include "Triangle.h"
#include <glm/simd/geometric.h>

using namespace glm;
namespace rfw::triangle
{
bool intersect(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt, const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2,
			   const float epsilon)
{
	const vec3 edge1 = p1 - p0;
	const vec3 edge2 = p2 - p0;

	const vec3 h = cross(dir, edge2);

	const float a = dot(edge1, h);
	if (a > -epsilon && a < epsilon)
		return false;

	const float f = 1.f / a;
	const vec3 s = org - p0;
	const float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
		return false;

	const vec3 q = cross(s, edge1);
	const float v = f * dot(dir, q);
	if (v < 0.0f || u + v > 1.0f)
		return false;

	const float t = f * dot(edge2, q);

	if (t > tmin && *rayt > t) // ray intersection
	{
		*rayt = t;
		return true;
	}

	return false;
}

bool intersect(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt, const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24,
			   const float epsilon)
{
	const vec3 p0 = p04;
	const vec3 p1 = p14;
	const vec3 p2 = p24;

	const vec3 edge1 = p1 - p0;
	const vec3 edge2 = p2 - p0;

	const vec3 h = cross(dir, edge2);

	const float a = dot(edge1, h);
	if (a > -epsilon && a < epsilon)
		return false;

	const float f = 1.f / a;
	const vec3 s = org - p0;
	const float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
		return false;

	const vec3 q = cross(s, edge1);
	const float v = f * dot(dir, q);
	if (v < 0.0f || u + v > 1.0f)
		return false;

	const float t = f * dot(edge2, q);

	if (t > tmin && *rayt > t) // ray intersection
	{
		*rayt = t;
		return true;
	}

	return false;
}

bool intersect_opt(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt, const glm::vec3 &p0, const glm::vec3 &e1, const glm::vec3 &e2)
{
	const vec3 h = cross(dir, e2);
	const float a = dot(e1, h);
	if (a > -EPSILON_TRIANGLE && a < EPSILON_TRIANGLE)
		return false;

	const float f = 1.f / a;
	const vec3 s = org - p0;
	const float u = f * dot(s, h);
	if (u < 0.0f || u > 1.0f)
		return false;

	const vec3 q = cross(s, e1);
	const float v = f * dot(dir, q);
	if (v < 0.0f || u + v > 1.0f)
		return false;

	const float t = f * dot(e2, q);

	if (t > tmin && *rayt > t) // ray intersection
	{
		*rayt = t;
		return true;
	}

	return false;
}

glm::vec3 getBaryCoords(const glm::vec3 &p, const glm::vec3 &normal, const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2)
{
	const float areaABC = glm::dot(normal, cross(p1 - p0, p2 - p0));
	const float areaPBC = glm::dot(normal, cross(p1 - p, p2 - p));
	const float areaPCA = glm::dot(normal, cross(p2 - p, p0 - p));

	const float alpha = areaPBC / areaABC;
	const float beta = areaPCA / areaABC;
	const float gamma = 1.f - alpha - beta;

	return vec3(alpha, beta, gamma);
}

glm::vec3 getBaryCoords(const glm::vec3 &p, const glm::vec3 &normal, const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	const float areaABC = glm::dot(normal, cross(p1 - p0, p2 - p0));
	const float areaPBC = glm::dot(normal, cross(p1 - p, p2 - p));
	const float areaPCA = glm::dot(normal, cross(p2 - p, p0 - p));

	const float alpha = areaPBC / areaABC;
	const float beta = areaPCA / areaABC;
	const float gamma = 1.f - alpha - beta;

	return vec3(alpha, beta, gamma);
}

int intersect4(cpurt::RayPacket4 &packet, const glm::vec3 &p0, const glm::vec3 &edge1, const glm::vec3 &edge2, __m128 *store_mask, float epsilon)
{
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128 zero4 = _mm_setzero_ps();

	const __m128 origin_x = _mm_load_ps(packet.origin_x);
	const __m128 origin_y = _mm_load_ps(packet.origin_y);
	const __m128 origin_z = _mm_load_ps(packet.origin_z);

	const __m128 direction_x = _mm_load_ps(packet.direction_x);
	const __m128 direction_y = _mm_load_ps(packet.direction_y);
	const __m128 direction_z = _mm_load_ps(packet.direction_z);

	const __m128 p0_x = _mm_set1_ps(p0.x);
	const __m128 p0_y = _mm_set1_ps(p0.y);
	const __m128 p0_z = _mm_set1_ps(p0.z);

	const __m128 edge1_x = _mm_set1_ps(edge1.x);
	const __m128 edge1_y = _mm_set1_ps(edge1.y);
	const __m128 edge1_z = _mm_set1_ps(edge1.z);

	const __m128 edge2_x = _mm_set1_ps(edge2.x);
	const __m128 edge2_y = _mm_set1_ps(edge2.y);
	const __m128 edge2_z = _mm_set1_ps(edge2.z);

	// Cross product
	// x = (ay * bz - az * by)
	// y = (az * bx - ax * bz)
	// z = (ax * by - ay * bx)
	// const vec3 h = cross(dir, edge2);
	const __m128 h_x4 = _mm_sub_ps(_mm_mul_ps(direction_y, edge2_z), _mm_mul_ps(direction_z, edge2_y));
	const __m128 h_y4 = _mm_sub_ps(_mm_mul_ps(direction_z, edge2_x), _mm_mul_ps(direction_x, edge2_z));
	const __m128 h_z4 = _mm_sub_ps(_mm_mul_ps(direction_x, edge2_y), _mm_mul_ps(direction_y, edge2_x));

	// const float a = dot(edge1, h);
	const __m128 a4 = _mm_add_ps(_mm_mul_ps(edge1_x, h_x4), _mm_add_ps(_mm_mul_ps(edge1_y, h_y4), _mm_mul_ps(edge1_z, h_z4)));

	// if (a > -epsilon && a < epsilon)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_a4 = _mm_or_ps(_mm_cmple_ps(a4, _mm_set1_ps(-epsilon)), _mm_cmpge_ps(a4, _mm_set1_ps(epsilon)));
	*store_mask = mask_a4;
	int mask = _mm_movemask_ps(mask_a4);
	if (mask == 0)
		return 0;

	// const float f = 1.f / a;
	const __m128 f4 = _mm_div_ps(one4, a4);

	// const vec3 s = org - p0;
	const __m128 s_x4 = _mm_sub_ps(origin_x, p0_x);
	const __m128 s_y4 = _mm_sub_ps(origin_y, p0_y);
	const __m128 s_z4 = _mm_sub_ps(origin_z, p0_z);

	// const float u = f * dot(s, h);
	const __m128 u4 = _mm_mul_ps(f4, _mm_add_ps(_mm_mul_ps(s_x4, h_x4), _mm_add_ps(_mm_mul_ps(s_y4, h_y4), _mm_mul_ps(s_z4, h_z4))));

	// if (u < 0.0f || u > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_u4 = _mm_and_ps(_mm_cmpgt_ps(u4, zero4), _mm_cmplt_ps(u4, one4));
	*store_mask = _mm_and_ps(*store_mask, mask_u4);
	mask = _mm_movemask_ps(mask_u4);
	if (mask == 0)
		return 0;

	// const vec3 q = cross(s, edge1);
	const __m128 q_x4 = _mm_sub_ps(_mm_mul_ps(s_y4, edge1_z), _mm_mul_ps(s_z4, edge1_y));
	const __m128 q_y4 = _mm_sub_ps(_mm_mul_ps(s_z4, edge1_x), _mm_mul_ps(s_x4, edge1_z));
	const __m128 q_z4 = _mm_sub_ps(_mm_mul_ps(s_x4, edge1_y), _mm_mul_ps(s_y4, edge1_x));

	// const float v = f * dot(dir, q);
	const __m128 dir_dot_q4 = _mm_add_ps(_mm_mul_ps(direction_x, q_x4), _mm_add_ps(_mm_mul_ps(direction_y, q_y4), _mm_mul_ps(direction_z, q_z4)));
	const __m128 v4 = _mm_mul_ps(f4, dir_dot_q4);

	// if (v < 0.0f || u + v > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_vu4 = _mm_and_ps(_mm_cmpge_ps(v4, zero4), _mm_cmple_ps(_mm_add_ps(u4, v4), one4));
	*store_mask = _mm_and_ps(*store_mask, mask_vu4);
	mask = _mm_movemask_ps(mask_vu4);
	if (mask == 0)
		return 0;

	// const float t = f * dot(edge2, q);
	const __m128 t4 = _mm_mul_ps(f4, _mm_add_ps(_mm_mul_ps(edge2_x, q_x4), _mm_add_ps(_mm_mul_ps(edge2_y, q_y4), _mm_mul_ps(edge2_z, q_z4))));

	// if (t > tmin && *rayt > t) // ray intersection
	*store_mask = _mm_and_ps(*store_mask, _mm_and_ps(_mm_cmpgt_ps(t4, _mm_set1_ps(1e-6f)), _mm_cmplt_ps(t4, _mm_load_ps(packet.t))));
	const int storage_mask = _mm_movemask_ps(*store_mask);
	if (storage_mask > 0)
	{
		//	*rayt = t;
		_mm_maskstore_ps(packet.t, _mm_castps_si128(*store_mask), t4);
		return storage_mask;
	}

	return 0;
}

int intersect8(cpurt::RayPacket8 &packet, const glm::vec3 &p0, const glm::vec3 &p14, const glm::vec3 &p24, float epsilon)
{
	// TODO
	// const vec3 p0 = p04;
	// const vec3 p1 = p14;
	// const vec3 p2 = p24;

	// const vec3 edge1 = p1 - p0;
	// const vec3 edge2 = p2 - p0;

	// const vec3 h = cross(dir, edge2);

	// const float a = dot(edge1, h);
	// if (a > -epsilon && a < epsilon)
	//	return false;

	// const float f = 1.f / a;
	// const vec3 s = org - p0;
	// const float u = f * dot(s, h);
	// if (u < 0.0f || u > 1.0f)
	//	return false;

	// const vec3 q = cross(s, edge1);
	// const float v = f * dot(dir, q);
	// if (v < 0.0f || u + v > 1.0f)
	//	return false;

	// const float t = f * dot(edge2, q);

	// if (t > tmin && *rayt > t) // ray intersection
	//{
	//	*rayt = t;
	//	return true;
	//}

	return 0;
}

AABB getBounds(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2)
{
	const vec3 mi = glm::min(p0, glm::min(p1, p2));
	const vec3 ma = glm::max(p0, glm::max(p1, p2));

	auto aabb = AABB();
	for (int i = 0; i < 3; i++)
	{
		aabb.bmin[i] = mi[i] - 1e-5f;
		aabb.bmax[i] = ma[i] + 1e-5f;
	}

	return aabb;
}

AABB getBounds(const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	const vec3 mi = glm::min(p0, glm::min(p1, p2));
	const vec3 ma = glm::max(p0, glm::max(p1, p2));

	auto aabb = AABB();
	for (int i = 0; i < 3; i++)
	{
		aabb.bmin[i] = mi[i] - 1e-5f;
		aabb.bmax[i] = ma[i] + 1e-5f;
	}

	return aabb;
}

glm::vec3 getRandomPointOnSurface(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2, float r1, float r2)
{
	if (r1 + r2 > 1.0f)
	{
		r1 = 1.0f - r1;
		r2 = 1.0f - r2;
	}

	const float a = 1.0f - r1 - r2;
	const float &b = r1;
	const float &c = r2;

	return a * p0 + b * p1 + c * p2;
}

glm::vec3 getRandomPointOnSurface(const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24, float r1, float r2)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	if (r1 + r2 > 1.0f)
	{
		r1 = 1.0f - r1;
		r2 = 1.0f - r2;
	}

	const float a = 1.0f - r1 - r2;
	const float &b = r1;
	const float &c = r2;

	return a * p0 + b * p1 + c * p2;
}

glm::vec3 getFaceNormal(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2) { return glm::normalize(glm::cross(p1 - p0, p2 - p0)); }

glm::vec3 getFaceNormal(const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);
	return glm::normalize(glm::cross(p1 - p0, p2 - p0));
}

glm::vec3 getNormal(const glm::vec3 &bary, const glm::vec3 &n0, const glm::vec3 &n1, const glm::vec3 &n2) { return bary.x * n0 + bary.y * n1 + bary.z * n2; }

glm::vec2 getTexCoords(const glm::vec3 &bary, const glm::vec2 &t0, const glm::vec2 &t1, const glm::vec2 &t2) { return bary.x * t0 + bary.y * t1 + bary.z * t2; }

float getArea(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2)
{
	const float a = glm::length(p0 - p1);
	const float b = glm::length(p1 - p2);
	const float c = glm::length(p2 - p0);
	const float s = (a + b + c) / 2.f;
	return sqrtf(s * (s - a) * (s - b) * (s - c));
}

float getArea(const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	const float a = glm::length(p0 - p1);
	const float b = glm::length(p1 - p2);
	const float c = glm::length(p2 - p0);
	const float s = (a + b + c) / 2.f;
	return sqrtf(s * (s - a) * (s - b) * (s - c));
}
} // namespace rfw::triangle