#include "../PCH.h"

using namespace glm;

namespace rfw::bvh
{

BVHNode::BVHNode()
{
	set_left_first(-1);
	set_count(-1);
}

BVHNode::BVHNode(int leftFirst, int count, AABB bounds) : bounds(bounds)
{
	set_left_first(leftFirst);
	set_count(-1);
}

bool BVHNode::intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t_min, float *t_max, const float min_t) const
{
	return bounds.intersect(org, dirInverse, t_min, t_max, min_t);
}

AABB BVHNode::refit(BVHNode *bvhTree, unsigned *primIDs, AABB *aabbs)
{
	if (is_leaf() && get_count() <= 0)
		return AABB();

	if (is_leaf())
	{
		AABB newBounds = {vec3(1e34f), vec3(-1e34f)};
		for (int idx = 0; idx < bounds.count; idx++)
			newBounds.grow(aabbs[primIDs[bounds.leftFirst + idx]]);

		bounds.offset_by(1e-5f);
		bounds.set_bounds(newBounds);
		return newBounds;
	}

	// BVH node
	auto new_bounds = AABB();
	new_bounds.grow(bvhTree[get_left_first()].refit(bvhTree, primIDs, aabbs));
	new_bounds.grow(bvhTree[get_left_first() + 1].refit(bvhTree, primIDs, aabbs));

	bounds.offset_by(1e-5f);
	bounds.set_bounds(new_bounds);
	return new_bounds;
}

void BVHNode::calculate_bounds(const AABB *aabbs, const unsigned int *primitiveIndices)
{
	auto new_bounds = AABB();
	for (auto idx = 0; idx < bounds.count; idx++)
		new_bounds.grow(aabbs[primitiveIndices[bounds.leftFirst + idx]]);

	bounds.offset_by(1e-5f);
	bounds.set_bounds(new_bounds);
}

bool BVHNode::traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						   const unsigned int *primIndices, const glm::vec3 *vertices, const glm::uvec3 *indices)
{
	return traverse_bvh(org, dir, t_min, t, hit_idx, nodes, primIndices, [&](int primID) {
		const auto idx = indices[primID];
		return rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z], 1e-5f);
	});
}

bool BVHNode::traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes,
						   const unsigned int *primIndices, const glm::vec3 *vertices)
{
	return traverse_bvh(org, dir, t_min, t, hit_idx, nodes, primIndices, [&](int primID) {
		const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
		return rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						   const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	return traverse_bvh(org, dir, t_min, t, hit_idx, nodes, primIndices, [&](int primID) {
		const auto idx = indices[primID];
		return rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z], 1e-5f);
	});
}

bool BVHNode::traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						   const glm::vec4 *vertices)
{
	return traverse_bvh(org, dir, t_min, t, hit_idx, nodes, primIndices, [&](int primID) {
		const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
		return rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const BVHNode *nodes, const unsigned *primIndices,
						   const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	return traverse_bvh(org, dir, t_min, t, hit_idx, nodes, primIndices,
						[&](int primID) { return rfw::triangle::intersect_opt(org, dir, t_min, t, p0s[primID], edge1s[primID], edge2s[primID]); });
}

bool BVHNode::traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *vertices, const glm::uvec3 *indices)
{
	return traverse_bvh_shadow(org, dir, t_min, maxDist, nodes, primIndices, [&](int primID) {
		const auto idx = indices[primID];
		return rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned int *primIndices,
								  const glm::vec3 *vertices)
{
	return traverse_bvh_shadow(org, dir, t_min, maxDist, nodes, primIndices, [&](int primID) {
		const auto idx = uvec3(primIndices[primID] * 3) + uvec3(0, 1, 2);
		return rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								  const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	return traverse_bvh_shadow(org, dir, t_min, maxDist, nodes, primIndices, [&](int primID) {
		const auto idx = indices[primID];
		return rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								  const glm::vec4 *vertices)
{
	return traverse_bvh_shadow(org, dir, t_min, maxDist, nodes, primIndices, [&](int primID) {
		const auto idx = uvec3(primIndices[primID] * 3) + uvec3(0, 1, 2);
		return rfw::triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
	});
}

bool BVHNode::traverse_bvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const BVHNode *nodes, const unsigned *primIndices,
								  const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	return traverse_bvh_shadow(org, dir, t_min, maxDist, nodes, primIndices, [&](int primID) {
		return rfw::triangle::intersect_opt(org, dir, t_min, &maxDist, p0s[primID], edge1s[primID], edge2s[primID]);
	});
}

} // namespace rfw::bvh