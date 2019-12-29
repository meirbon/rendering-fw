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

AABB getBounds(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2)
{
	const vec3 min = glm::min(p0, glm::min(p1, p2));
	const vec3 max = glm::max(p0, glm::max(p1, p2));

	return {min - 0.001f, max + 0.01f};
}
AABB getBounds(const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	const vec3 min = glm::min(p0, glm::min(p1, p2));
	const vec3 max = glm::max(p0, glm::max(p1, p2));

	return {min - 0.001f, max + 0.01f};
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