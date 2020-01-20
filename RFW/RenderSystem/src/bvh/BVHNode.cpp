#include "../rfw.h"

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

AABB BVHNode::refit(BVHNode *bvhTree, uint *primIDs, AABB *aabbs)
{
	if (is_leaf() && get_count() <= 0)
		return AABB::invalid();

	if (is_leaf())
	{
		AABB newBounds = {vec3(1e34f), vec3(-1e34f)};
		for (int idx = 0; idx < bounds.count; idx++)
			newBounds.grow(aabbs[primIDs[bounds.leftFirst + idx]]);

		bounds.offset_by(1e-6f);
		bounds.set_bounds(newBounds);
		return newBounds;
	}

	// BVH node
	auto new_bounds = AABB();
	new_bounds.grow(bvhTree[get_left_first()].refit(bvhTree, primIDs, aabbs));
	new_bounds.grow(bvhTree[get_left_first() + 1].refit(bvhTree, primIDs, aabbs));

	bounds.offset_by(1e-6f);
	bounds.set_bounds(new_bounds);
	return new_bounds;
}

void BVHNode::calculate_bounds(const AABB *aabbs, const unsigned int *primitiveIndices)
{
	auto new_bounds = AABB();
	for (auto idx = 0; idx < bounds.count; idx++)
		new_bounds.grow(aabbs[primitiveIndices[bounds.leftFirst + idx]]);

	bounds.offset_by(1e-6f);
	bounds.set_bounds(new_bounds);
}

} // namespace rfw::bvh