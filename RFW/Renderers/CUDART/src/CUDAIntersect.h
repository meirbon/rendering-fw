#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "BVH/BVHNode.h"
#include "BVH/MBVHNode.h"

inline __host__ __device__ bool intersect_triangle(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt,
												   const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24,
												   const float epsilon = 1e-8f)
{
	const vec3 p0 = p04;

	const vec3 e1 = vec3(p14) - p0;
	const vec3 e2 = vec3(p24) - p0;

	const vec3 h = cross(dir, e2);

	const float a = dot(e1, h);
	if (a > -epsilon && a < epsilon)
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

inline __host__ __device__ bool intersect_triangle(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt,
												   const glm::vec4 &p04, const glm::vec4 &p14, const glm::vec4 &p24,
												   glm::vec2 *bary, const float epsilon = 1e-8f)
{
	const vec3 p0 = p04;
	const vec3 p1 = p14;
	const vec3 p2 = p24;

	const vec3 e1 = p1 - p0;
	const vec3 e2 = p2 - p0;

	const vec3 h = cross(dir, e2);

	const float a = dot(e1, h);
	if (a > -epsilon && a < epsilon)
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
		// Barycentrics
		const vec3 p = org + t * dir;
		const vec3 N = normalize(cross(e1, e2));
		const float areaABC = dot(N, cross(e1, e2));
		const float areaPBC = dot(N, cross(p1 - p, p2 - p));
		const float areaPCA = dot(N, cross(p2 - p, p0 - p));
		*bary = glm::vec2(areaPBC / areaABC, areaPCA / areaABC);

		*rayt = t;
		return true;
	}

	return false;
}

inline __host__ __device__ glm::vec2 get_barycentrics(const glm::vec3 &p, const glm::vec3 &normal, const glm::vec4 &p04,
													  const glm::vec4 &p14, const glm::vec4 &p24)
{
	const vec3 p0 = vec3(p04);
	const vec3 p1 = vec3(p14);
	const vec3 p2 = vec3(p24);

	const float areaABC = dot(normal, cross(p1 - p0, p2 - p0));
	const float areaPBC = dot(normal, cross(p1 - p, p2 - p));
	const float areaPCA = dot(normal, cross(p2 - p, p0 - p));

	const float alpha = areaPBC / areaABC;
	const float beta = areaPCA / areaABC;

	return vec2(alpha, beta);
}

inline __host__ __device__ bool intersect_node(const rfw::bvh::AABB &aabb, const glm::vec3 &org,
											   const glm::vec3 &dirInverse, float *tmin, float *tmax, float t)
{
	const float tx1 = (aabb.bmin[0] - org.x) * dirInverse.x;
	const float tx2 = (aabb.bmax[0] - org.x) * dirInverse.x;

	*tmin = glm::min(tx1, tx2);
	*tmax = glm::max(tx1, tx2);

	const float ty1 = (aabb.bmin[1] - org.y) * dirInverse.y;
	const float ty2 = (aabb.bmax[1] - org.y) * dirInverse.y;

	*tmin = glm::max(*tmin, glm::min(ty1, ty2));
	*tmax = glm::min(*tmax, glm::max(ty1, ty2));

	const float tz1 = (aabb.bmin[2] - org.z) * dirInverse.z;
	const float tz2 = (aabb.bmax[2] - org.z) * dirInverse.z;

	*tmin = glm::max(*tmin, glm::min(tz1, tz2));
	*tmax = glm::min(*tmax, glm::max(tz1, tz2));

	return (*tmax) > (*tmin) && (*tmin) < t;
}

template <typename T> inline __host__ __device__ void swap(T &a, T &b)
{
	T temp = a;
	a = b;
	b = temp;
}

struct MBVHHit
{
	union {
		glm::vec4 tmin;
		glm::ivec4 tmini;
	};

	bvec4 result;
};

__host__ __device__ MBVHHit intersect_quad_node(const rfw::bvh::MBVHNode &node, const glm::vec3 &org,
												const glm::vec3 &dirInverse, float t)
{
	MBVHHit hit;

	vec4 t1 = (node.bminx4 - org.x) * dirInverse.x;
	vec4 t2 = (node.bmaxx4 - org.x) * dirInverse.x;

	hit.tmin = min(t1, t2);
	glm::vec4 tmax = max(t1, t2);

	t1 = (node.bminy4 - org.y) * dirInverse.y;
	t2 = (node.bmaxy4 - org.y) * dirInverse.y;

	hit.tmin = max(hit.tmin, min(t1, t2));
	tmax = glm::min(tmax, glm::max(t1, t2));

	t1 = (node.bminz4 - org.z) * dirInverse.z;
	t2 = (node.bmaxz4 - org.z) * dirInverse.z;

	hit.tmin = max(hit.tmin, min(t1, t2));
	tmax = min(tmax, max(t1, t2));

	hit.result = greaterThanEqual(tmax, hit.tmin) && lessThan(hit.tmin, glm::vec4(t));

	hit.tmini[0] = ((hit.tmini[0] & 0xFFFFFFFCu) | 0b00u);
	hit.tmini[1] = ((hit.tmini[1] & 0xFFFFFFFCu) | 0b01u);
	hit.tmini[2] = ((hit.tmini[2] & 0xFFFFFFFCu) | 0b10u);
	hit.tmini[3] = ((hit.tmini[3] & 0xFFFFFFFCu) | 0b11u);

	if (hit.tmin[0] > hit.tmin[1])
		swap<float>(hit.tmin[0], hit.tmin[1]);
	if (hit.tmin[2] > hit.tmin[3])
		swap<float>(hit.tmin[2], hit.tmin[3]);
	if (hit.tmin[0] > hit.tmin[2])
		swap<float>(hit.tmin[0], hit.tmin[2]);
	if (hit.tmin[1] > hit.tmin[3])
		swap<float>(hit.tmin[1], hit.tmin[3]);
	if (hit.tmin[2] > hit.tmin[3])
		swap<float>(hit.tmin[2], hit.tmin[3]);

	return hit;
}

template <typename FUNC>
__host__ __device__ bool intersect_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx,
									   const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices,
									   const FUNC &intersection)
{
	bool valid = false;
	rfw::bvh::BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const uint primID = primIndices[node.get_left_first() + i];
				if (intersection(primID))
				{
					valid = true;
					*hit_idx = primID;
				}
			}
		}
		else
		{
			const bool hit_left =
				intersect_node(nodes[node.get_left_first()].bounds, org, dirInverse, &tNear1, &tFar1, *t);
			const bool hit_right =
				intersect_node(nodes[node.get_left_first() + 1].bounds, org, dirInverse, &tNear2, &tFar2, *t);

			if (hit_left && hit_right)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
			}
			else if (hit_left)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hit_right)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return valid;
}

template <typename FUNC>
__host__ __device__ bool intersect_mbvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx,
										const rfw::bvh::MBVHNode *nodes, const unsigned int *primIndices,
										const FUNC &intersection)
{
	bool valid = false;
	rfw::bvh::MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count >= 0) // leaf node
		{
			for (int i = 0; i < count; i++)
			{
				const uint primID = primIndices[leftFirst + i];
				if (intersection(primID))
				{
					valid = true;
					*hit_idx = primID;
				}
			}
			continue;
		}

		const MBVHHit hit = intersect_quad_node(nodes[leftFirst], org, dirInverse, *t);
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
__host__ __device__ bool intersect_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float t_max,
											  const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices,
											  const FUNC &intersection)
{
	rfw::bvh::BVHTraversal todo[32];
	int stackPtr = 0;
	float tNear1, tFar1;
	float tNear2, tFar2;

	const glm::vec3 dirInverse = 1.0f / dir;

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const uint primID = primIndices[node.get_left_first() + i];
				if (intersection(primID))
					return true;
			}
		}
		else
		{
			const bool hit_left =
				intersect_node(nodes[node.get_left_first()].bounds, org, dirInverse, &tNear1, &tFar1, t_max);
			const bool hit_right =
				intersect_node(nodes[node.get_left_first() + 1].bounds, org, dirInverse, &tNear2, &tFar2, t_max);

			if (hit_left && hit_right)
			{
				if (tNear1 < tNear2)
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
				}
				else
				{
					stackPtr++;
					todo[stackPtr] = {node.get_left_first() + 1};
					stackPtr++;
					todo[stackPtr] = {node.get_left_first()};
				}
			}
			else if (hit_left)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hit_right)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return false;
}

template <typename FUNC>
__host__ __device__ bool intersect_mbvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float t_max,
											   const rfw::bvh::MBVHNode *nodes, const unsigned int *primIndices,
											   const FUNC &intersection)
{
	rfw::bvh::MBVHTraversal todo[32];
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
				if (intersection(primID))
					return true;
			}
			continue;
		}

		const MBVHHit hit = intersect_quad_node(nodes[leftFirst], org, dirInverse, t_max);
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

	return false;
}