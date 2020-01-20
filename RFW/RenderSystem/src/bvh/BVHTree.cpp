#include "../rfw.h"

using namespace glm;
using namespace rfw;

namespace rfw::bvh
{

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount) : vertex_count(vertexCount), face_count(vertexCount / 3)
{
	indices = nullptr;
	pool_ptr.store(0);
	reset();
	set_vertices(vertices);
}

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount) : vertex_count(vertexCount), face_count(faceCount)
{
	pool_ptr.store(0);
	reset();
	set_vertices(vertices, indices);
}

void BVHTree::construct_bvh(bool printBuildTime)
{
	assert(vertices);

	if (face_count > 0)
	{
		utils::Timer t = {};
		pool_ptr = 2;
		bvh_nodes.emplace_back();
		bvh_nodes.emplace_back();

		auto &rootNode = bvh_nodes[0];
		rootNode.bounds.leftFirst = 0;
		rootNode.bounds.count = static_cast<int>(face_count);
		rootNode.calculate_bounds(aabbs.data(), prim_indices.data());

		// rootNode.Subdivide(m_AABBs.data(), m_BVHPool.data(), m_PrimitiveIndices.data(), 1, m_PoolPtr);
		rootNode.subdivide_mt<7, 64, 5>(aabbs.data(), bvh_nodes.data(), prim_indices.data(), building_threads, 1, pool_ptr);

		if (pool_ptr > 2)
		{
			rootNode.bounds.count = -1;
			rootNode.set_left_first(2);
		}
		else
		{
			rootNode.bounds.count = static_cast<int>(face_count);
			rootNode.set_left_first(0);
		}

		bvh_nodes.resize(pool_ptr);

		if (printBuildTime)
			std::cout << "Building BVH took: " << t.elapsed() << " ms for " << face_count << " triangles. Poolptr: " << pool_ptr << std::endl;
	}
}

void BVHTree::reset()
{
	bvh_nodes.clear();
	if (face_count > 0)
	{
		prim_indices.clear();
		prim_indices.reserve(face_count);
		for (int i = 0; i < face_count; i++)
			prim_indices.push_back(i);
		bvh_nodes.resize(face_count * 2);
	}
}

void BVHTree::refit(const glm::vec4 *vertices)
{
	set_vertices(vertices);
	aabb = bvh_nodes[0].refit(bvh_nodes.data(), prim_indices.data(), aabbs.data());
}

void BVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	set_vertices(vertices, indices);
	aabb = bvh_nodes[0].refit(bvh_nodes.data(), prim_indices.data(), aabbs.data());
}

bool BVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx)
{
	return BVHNode::traverse_bvh(origin, dir, t_min, ray_t, primIdx, bvh_nodes.data(), prim_indices.data(), [&](uint primID) {
		const vec3 &p0 = p0s[primID];
		const vec3 &e1 = edge1s[primID];
		const vec3 &e2 = edge2s[primID];
		const vec3 h = cross(dir, e2);

		const float a = dot(e1, h);
		if (a > -1e-6f && a < 1e-6f)
			return false;

		const float f = 1.f / a;
		const vec3 s = origin - p0;
		const float u = f * dot(s, h);
		if (u < 0.0f || u > 1.0f)
			return false;

		const vec3 q = cross(s, e1);
		const float v = f * dot(dir, q);
		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float t = f * dot(e2, q);

		if (t > t_min && *ray_t > t) // ray intersection
		{
			*ray_t = t;
			return true;
		}

		return false;
	});
}

bool BVHTree::traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max)
{
	return BVHNode::traverse_bvh_shadow(origin, dir, t_min, t_max, bvh_nodes.data(), prim_indices.data(), [&](uint primID) {
		const vec3 &p0 = p0s[primID];
		const vec3 &e1 = edge1s[primID];
		const vec3 &e2 = edge2s[primID];

		const vec3 h = cross(dir, e2);

		const float a = dot(e1, h);
		if (a > -1e-6f && a < 1e-6f)
			return false;

		const float f = 1.f / a;
		const vec3 s = origin - p0;
		const float u = f * dot(s, h);
		if (u < 0.0f || u > 1.0f)
			return false;

		const vec3 q = cross(s, e1);
		const float v = f * dot(dir, q);
		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float t = f * dot(e2, q);

		if (t > t_min && t_max > t) // ray intersection
			return true;

		return false;
	});
}

void BVHTree::set_vertices(const glm::vec4 *vertices)
{
	vertices = vertices;
	indices = nullptr;

	aabb = AABB::invalid();

	// Recalculate data
	aabbs.resize(face_count);
	p0s.resize(face_count);
	edge1s.resize(face_count);
	edge2s.resize(face_count);

	for (int i = 0; i < face_count; i++)
	{
		const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);

		const vec3 p0 = vec3(vertices[idx.x]);
		const vec3 p1 = vec3(vertices[idx.y]);
		const vec3 p2 = vec3(vertices[idx.z]);

		const vec3 mi = glm::min(p0, glm::min(p1, p2));
		const vec3 ma = glm::max(p0, glm::max(p1, p2));

		for (int j = 0; j < 3; j++)
		{
			aabbs[i].bmin[j] = mi[j] - 1e-6f;
			aabbs[i].bmax[j] = ma[j] + 1e-6f;
		}

		aabb.grow(aabbs[i]);

		p0s[i] = p0;
		edge1s[i] = p1 - p0;
		edge2s[i] = p2 - p0;
	}
}

void BVHTree::set_vertices(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	vertices = vertices;
	indices = indices;

	aabb = AABB::invalid();

	// Recalculate data
	aabbs.resize(face_count);
	p0s.resize(face_count);
	edge1s.resize(face_count);
	edge2s.resize(face_count);

	for (int i = 0; i < face_count; i++)
	{
		const uvec3 &idx = indices[i];

		const vec3 p0 = vec3(vertices[idx.x]);
		const vec3 p1 = vec3(vertices[idx.y]);
		const vec3 p2 = vec3(vertices[idx.z]);

		const vec3 mi = glm::min(p0, glm::min(p1, p2));
		const vec3 ma = glm::max(p0, glm::max(p1, p2));

		for (int j = 0; j < 3; j++)
		{
			aabbs[i].bmin[j] = mi[j] - 1e-6f;
			aabbs[i].bmax[j] = ma[j] + 1e-6f;
		}

		aabb.grow(aabbs[i]);

		p0s[i] = p0;
		edge1s[i] = p1 - p0;
		edge2s[i] = p2 - p0;
	}
}
} // namespace rfw::bvh