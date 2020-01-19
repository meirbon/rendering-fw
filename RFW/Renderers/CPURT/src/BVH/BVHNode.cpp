#include "../PCH.h"

using namespace glm;

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

bool BVHNode::intersect(cpurt::RayPacket4 &packet4, __m128 *tmin_4, __m128 *tmax_4, const float min_t) const
{
	return bounds.intersect(packet4, tmin_4, tmax_4, min_t);
}

bool BVHNode::intersect(cpurt::RayPacket8 &packet8, float min_t) const
{
	static const __m256 one8 = _mm256_set1_ps(1.0f);

	// const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m256 origin_x8 = _mm256_load_ps(packet8.origin_x);
	const __m256 origin_y8 = _mm256_load_ps(packet8.origin_y);
	const __m256 origin_z8 = _mm256_load_ps(packet8.origin_z);

	// const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m256 inv_direction_x8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_x));
	const __m256 inv_direction_y8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_y));
	const __m256 inv_direction_z8 = _mm256_div_ps(one8, _mm256_load_ps(packet8.direction_z));

	// const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const __m256 t1_8_x = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[0]), origin_x8), inv_direction_x8);
	const __m256 t1_8_y = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[1]), origin_y8), inv_direction_y8);
	const __m256 t1_8_z = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmin[2]), origin_z8), inv_direction_z8);

	// const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;
	const __m256 t2_8_x = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[0]), origin_x8), inv_direction_x8);
	const __m256 t2_8_y = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[1]), origin_y8), inv_direction_y8);
	const __m256 t2_8_z = _mm256_mul_ps(_mm256_sub_ps(_mm256_set1_ps(bounds.bmax[2]), origin_z8), inv_direction_z8);

	// const glm::vec3 min = glm::min(t1, t2);
	const __m256 tmin_x8 = _mm256_min_ps(t1_8_x, t2_8_x);
	const __m256 tmin_y8 = _mm256_min_ps(t1_8_y, t2_8_y);
	const __m256 tmin_z8 = _mm256_min_ps(t1_8_z, t2_8_z);

	// const glm::vec3 max = glm::max(t1, t2);
	const __m256 tmax_x8 = _mm256_max_ps(t1_8_x, t2_8_x);
	const __m256 tmax_y8 = _mm256_max_ps(t1_8_y, t2_8_y);
	const __m256 tmax_z8 = _mm256_max_ps(t1_8_z, t2_8_z);

	//*t_min = glm::max(min.x, glm::max(min.y, min.z));
	const __m256 tmin_8 = _mm256_max_ps(tmin_x8, _mm256_max_ps(tmin_y8, tmin_z8));
	//*t_max = glm::min(max.x, glm::min(max.y, max.z));
	const __m256 tmax_8 = _mm256_min_ps(tmax_x8, _mm256_min_ps(tmax_y8, tmax_z8));

	const __m128 min_t4 = _mm_set1_ps(min_t);

	const __m128 tmin_4_0 = _mm256_extractf128_ps(tmin_8, 0);
	const __m128 tmin_4_1 = _mm256_extractf128_ps(tmin_8, 1);

	const __m128 tmax_4_0 = _mm256_extractf128_ps(tmax_8, 0);
	const __m128 tmax_4_1 = _mm256_extractf128_ps(tmax_8, 1);

	const __m128 mask4_0 = _mm_and_ps(_mm_cmpge_ps(tmax_4_0, min_t4), _mm_cmplt_ps(tmin_4_0, tmax_4_0));
	const __m128 mask4_1 = _mm_and_ps(_mm_cmpge_ps(tmax_4_1, min_t4), _mm_cmplt_ps(tmin_4_1, tmax_4_1));

	// return *t_max >= min_t && *t_min < *t_max;
	return _mm_movemask_ps(mask4_0) > 0 || _mm_movemask_ps(mask4_1) > 0;
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

int BVHNode::traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned *primIndices, const glm::vec3 *p0s,
						   const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask)
{
	bool valid = false;
	BVHTraversal todo[32];
	int stackPtr = 0;
	int hitMask = 0;
	__m128 tNear1, tFar1;
	__m128 tNear2, tFar2;

	__m128 store_mask = _mm_setzero_ps();

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const auto primIdx = primIndices[node.get_left_first() + i];
				const auto idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);
				const int mask = rfw::triangle::intersect4(packet, p0s[primIdx], edge1s[primIdx], edge2s[primIdx], &store_mask);
				if (mask != 0)
				{
					hitMask |= mask;
					*hit_mask = _mm_or_ps(*hit_mask, store_mask);
					_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
				}
			}
		}
		else
		{
			const bool hitLeft = nodes[node.get_left_first()].intersect(packet, &tNear1, &tFar1, t_min);
			const bool hitRight = nodes[node.get_left_first() + 1].intersect(packet, &tNear2, &tFar2, t_min);

			if (hitLeft && hitRight)
			{
				if (_mm_movemask_ps(_mm_cmplt_ps(tNear1, tNear2)) > 0 /* tNear1 < tNear2*/)
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
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return hitMask;
}

int BVHNode::traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned *primIndices, const glm::vec4 *vertices,
						   __m128 *hit_mask)
{
	bool valid = false;
	BVHTraversal todo[32];
	int stackPtr = 0;
	int hitMask = 0;
	__m128 tNear1, tFar1;
	__m128 tNear2, tFar2;

	__m128 store_mask = _mm_setzero_ps();

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const auto primIdx = primIndices[node.get_left_first() + i];
				const auto idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);
				const int mask = rfw::triangle::intersect4(packet, vertices[idx.x], vertices[idx.y], vertices[idx.z], &store_mask);
				if (mask != 0)
				{
					hitMask |= mask;
					*hit_mask = _mm_or_ps(*hit_mask, store_mask);
					_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
				}
			}
		}
		else
		{
			const bool hitLeft = nodes[node.get_left_first()].intersect(packet, &tNear1, &tFar1, t_min);
			const bool hitRight = nodes[node.get_left_first() + 1].intersect(packet, &tNear2, &tFar2, t_min);

			if (hitLeft && hitRight)
			{
				if (_mm_movemask_ps(_mm_cmplt_ps(tNear1, tNear2)) > 0 /* tNear1 < tNear2*/)
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
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return hitMask;
}

int BVHNode::traverse_bvh4(cpurt::RayPacket4 &packet, float t_min, const BVHNode *nodes, const unsigned *primIndices, const glm::vec4 *vertices,
						   const glm::uvec3 *indices, __m128 *hit_mask)
{
	bool valid = false;
	BVHTraversal todo[32];
	int stackPtr = 0;
	int hitMask = 0;
	__m128 tNear1, tFar1;
	__m128 tNear2, tFar2;

	__m128 store_mask = _mm_setzero_ps();

	todo[stackPtr].nodeIdx = 0;
	while (stackPtr >= 0)
	{
		const auto &node = nodes[todo[stackPtr].nodeIdx];
		stackPtr--;

		if (node.get_count() > -1)
		{
			for (int i = 0; i < node.get_count(); i++)
			{
				const auto primIdx = primIndices[node.get_left_first() + i];
				const auto idx = indices[primIdx];
				const int mask = rfw::triangle::intersect4(packet, vertices[idx.x], vertices[idx.y], vertices[idx.z], &store_mask);
				if (mask != 0)
				{
					hitMask |= mask;
					*hit_mask = _mm_or_ps(*hit_mask, store_mask);
					_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
				}
			}
		}
		else
		{
			const bool hitLeft = nodes[node.get_left_first()].intersect(packet, &tNear1, &tFar1, t_min);
			const bool hitRight = nodes[node.get_left_first() + 1].intersect(packet, &tNear2, &tFar2, t_min);

			if (hitLeft && hitRight)
			{
				if (_mm_movemask_ps(_mm_cmplt_ps(tNear1, tNear2)) > 0 /* tNear1 < tNear2*/)
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
			else if (hitLeft)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hitRight)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return hitMask;
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
		const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
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
		const auto idx = uvec3(primID * 3) + uvec3(0, 1, 2);
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
