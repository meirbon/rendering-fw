#include "../rfw.h"

using namespace glm;
using namespace rfw;

namespace rfw::bvh
{

#define EDGE_INTERSECTION 0

MBVHTree::MBVHTree(BVHTree *orgTree) { this->bvh = orgTree; }

void MBVHTree::construct_bvh(bool printBuildTime)
{
	mbvh_nodes.clear();
	// Worst case, this BVH becomes as big as the original
	mbvh_nodes.resize(bvh->bvh_nodes.size());
	if (bvh->aabbs.empty())
		return;

	utils::Timer t{};
	pool_ptr.store(1);
	MBVHNode &mRootNode = mbvh_nodes[0];

	if (bvh->pool_ptr <= 4) // Original tree first in single MBVH node
	{
		int num_children = 0;
		mRootNode.merge_node(bvh->bvh_nodes[0], bvh->bvh_nodes, num_children);
	}
	else
	{
		mRootNode.merge_nodes(bvh->bvh_nodes[0], bvh->bvh_nodes, mbvh_nodes.data(), pool_ptr);
	}

	mbvh_nodes.resize(pool_ptr);
	if (printBuildTime)
		std::cout << "Building MBVH took: " << t.elapsed() << " ms. Poolptr: " << pool_ptr.load() << std::endl;

#ifndef NDEBUG
	mbvh_nodes[0].validate(mbvh_nodes, bvh->prim_indices, pool_ptr, bvh->face_count);
#endif
}

void MBVHTree::refit(const glm::vec4 *vertices)
{
	bvh->refit(vertices);
	construct_bvh();
}

void MBVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	bvh->refit(vertices, indices);
	construct_bvh();
}

bool MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx)
{
	return MBVHNode::traverse_mbvh(origin, dir, t_min, ray_t, primIdx, mbvh_nodes.data(), bvh->prim_indices.data(), [&](uint primID) {
		const vec3 &p0 = bvh->p0s[primID];
		const vec3 &e1 = bvh->edge1s[primID];
		const vec3 &e2 = bvh->edge2s[primID];
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

bool MBVHTree::traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max)
{
	return MBVHNode::traverse_mbvh_shadow(origin, dir, t_min, t_max, mbvh_nodes.data(), bvh->prim_indices.data(), [&](uint primID) {
		const vec3 &p0 = bvh->p0s[primID];
		const vec3 &e1 = bvh->edge1s[primID];
		const vec3 &e2 = bvh->edge2s[primID];

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

} // namespace rfw::bvh