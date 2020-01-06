#define GLM_FORCE_AVX
#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"

#include <thread>
#include <future>

using namespace glm;
using namespace rfw;

void MBVHNode::set_bounds(unsigned int nodeIdx, const vec3 &min, const vec3 &max)
{
	this->bminx[nodeIdx] = min.x;
	this->bminy[nodeIdx] = min.y;
	this->bminz[nodeIdx] = min.z;

	this->bmaxx[nodeIdx] = max.x;
	this->bmaxy[nodeIdx] = max.y;
	this->bmaxz[nodeIdx] = max.z;
}

void MBVHNode::set_bounds(unsigned int nodeIdx, const AABB &bounds)
{
	this->bminx[nodeIdx] = bounds.xMin;
	this->bminy[nodeIdx] = bounds.yMin;
	this->bminz[nodeIdx] = bounds.zMin;

	this->bmaxx[nodeIdx] = bounds.xMax;
	this->bmaxy[nodeIdx] = bounds.yMax;
	this->bmaxz[nodeIdx] = bounds.zMax;
}

MBVHHit MBVHNode::intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t, const float t_min) const
{
#if 1
	static const __m128i mask = _mm_set1_epi32(0xFFFFFFFCu);
	static const __m128i or_mask = _mm_set_epi32(0b11, 0b10, 0b01, 0b00);

	MBVHHit hit{};

	__m128 orgComponent = _mm_set1_ps(org.x);
	__m128 dirComponent = _mm_set1_ps(dirInverse.x);

	__m128 t1 = _mm_mul_ps(_mm_sub_ps(bminx_4, orgComponent), dirComponent);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(bmaxx_4, orgComponent), dirComponent);

	hit.t_min = _mm_min_ps(t1, t2);
	__m128 t_max = _mm_max_ps(t1, t2);

	orgComponent = _mm_set1_ps(org.y);
	dirComponent = _mm_set1_ps(dirInverse.y);

	t1 = _mm_mul_ps(_mm_sub_ps(bminy_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxy_4, orgComponent), dirComponent);

	hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	orgComponent = _mm_set1_ps(org.z);
	dirComponent = _mm_set1_ps(dirInverse.z);

	t1 = _mm_mul_ps(_mm_sub_ps(bminz_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxz_4, orgComponent), dirComponent);

	hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	const __m128 greaterThan0 = _mm_cmpge_ps(t_max, _mm_set1_ps(t_min));
	const __m128 lessThanEqualMax = _mm_cmplt_ps(hit.t_min, t_max);
	const __m128 lessThanT = _mm_cmplt_ps(hit.t_min, _mm_set1_ps(*t));

	const __m128 result = _mm_and_ps(greaterThan0, _mm_and_ps(lessThanEqualMax, lessThanT));
	const int resultMask = _mm_movemask_ps(result);

	hit.t_mini = _mm_and_si128(hit.t_mini, mask);
	hit.t_mini = _mm_or_si128(hit.t_mini, or_mask);
	hit.result = bvec4(resultMask & 1, resultMask & 2, resultMask & 4, resultMask & 8);

	if (hit.tmin[0] > hit.tmin[1])
		std::swap(hit.tmin[0], hit.tmin[1]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);
	if (hit.tmin[0] > hit.tmin[2])
		std::swap(hit.tmin[0], hit.tmin[2]);
	if (hit.tmin[1] > hit.tmin[3])
		std::swap(hit.tmin[1], hit.tmin[3]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);

	return hit;
#else
	MBVHHit hit{};

	glm::vec4 t1 = (bminx4 - org.x) * dirInverse.x;
	glm::vec4 t2 = (bmaxx4 - org.x) * dirInverse.x;

	hit.tmin4 = glm::min(t1, t2);
	glm::vec4 tmax = glm::max(t1, t2);

	t1 = (bminy4 - org.y) * dirInverse.y;
	t2 = (bmaxy4 - org.y) * dirInverse.y;

	hit.tmin4 = glm::max(hit.tmin4, glm::min(t1, t2));
	tmax = glm::min(tmax, glm::max(t1, t2));

	t1 = (bminz4 - org.z) * dirInverse.z;
	t2 = (bmaxz4 - org.z) * dirInverse.z;

	hit.tmin4 = glm::max(hit.tmin4, glm::min(t1, t2));
	tmax = glm::min(tmax, glm::max(t1, t2));

	hit.tmini[0] = ((hit.tmini[0] & 0xFFFFFFFCu) | 0b00u);
	hit.tmini[1] = ((hit.tmini[1] & 0xFFFFFFFCu) | 0b01u);
	hit.tmini[2] = ((hit.tmini[2] & 0xFFFFFFFCu) | 0b10u);
	hit.tmini[3] = ((hit.tmini[3] & 0xFFFFFFFCu) | 0b11u);

	hit.result = greaterThan(tmax, glm::vec4(0.0f)) && lessThanEqual(hit.tmin4, tmax) && lessThanEqual(hit.tmin4, glm::vec4(*t));

	if (hit.tmin[0] > hit.tmin[1])
		std::swap(hit.tmin[0], hit.tmin[1]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);
	if (hit.tmin[0] > hit.tmin[2])
		std::swap(hit.tmin[0], hit.tmin[2]);
	if (hit.tmin[1] > hit.tmin[3])
		std::swap(hit.tmin[1], hit.tmin[3]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);

	return hit;
#endif
}

MBVHHit MBVHNode::intersect4(cpurt::RayPacket4 &packet, float min_t) const
{
	static const __m128i mask = _mm_set1_epi32(0xFFFFFFFCu);
	static const __m128i or_mask = _mm_set_epi32(0b11, 0b10, 0b01, 0b00);
	MBVHHit hit = {};

	for (int i = 0; i < 4; i++)
	{
#if 1
		const vec3 origin = vec3(packet.origin_x[i], packet.origin_y[i], packet.origin_z[i]);
		const vec3 dirInverse = 1.0f / vec3(packet.direction_x[i], packet.direction_y[i], packet.direction_z[i]);
		const auto temp_hit = intersect(origin, dirInverse, &packet.t[i], min_t);
		hit.result |= temp_hit.result;

		hit.t_min = _mm_min_ps(hit.t_min, temp_hit.t_min);
#else
		__m128 orgComponent = _mm_set1_ps(packet.origin_x[i]);
		__m128 dirComponent = _mm_set1_ps(1.0f / packet.direction_x[i]);

		__m128 t1 = _mm_mul_ps(_mm_sub_ps(bminx_4, orgComponent), dirComponent);
		__m128 t2 = _mm_mul_ps(_mm_sub_ps(bmaxx_4, orgComponent), dirComponent);

		__m128 t_min = _mm_min_ps(t1, t2);
		__m128 t_max = _mm_max_ps(t1, t2);

		orgComponent = _mm_set1_ps(packet.origin_y[i]);
		dirComponent = _mm_set1_ps(1.0f / packet.direction_y[i]);

		t1 = _mm_mul_ps(_mm_sub_ps(bminy_4, orgComponent), dirComponent);
		t2 = _mm_mul_ps(_mm_sub_ps(bmaxy_4, orgComponent), dirComponent);

		t_min = _mm_max_ps(t_min, _mm_min_ps(t1, t2));
		t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

		orgComponent = _mm_set1_ps(packet.origin_z[i]);
		dirComponent = _mm_set1_ps(1.0f / packet.direction_z[i]);

		t1 = _mm_mul_ps(_mm_sub_ps(bminz_4, orgComponent), dirComponent);
		t2 = _mm_mul_ps(_mm_sub_ps(bmaxz_4, orgComponent), dirComponent);

		t_min = _mm_max_ps(t_min, _mm_min_ps(t1, t2));
		t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

		const __m128 greaterThan0 = _mm_cmpgt_ps(t_max, _mm_set1_ps(min_t));
		const __m128 lessThanEqualMax = _mm_cmple_ps(t_min, t_max);
		const __m128 lessThanT = _mm_cmple_ps(t_min, _mm_set1_ps(packet.t[i]));

		const __m128 hit_4 = _mm_and_ps(greaterThan0, _mm_and_ps(lessThanEqualMax, lessThanT));
		const int resultMask = _mm_movemask_ps(hit_4);
		hit.result.x = hit.result.x || (resultMask & 1);
		hit.result.y = hit.result.y || (resultMask & 2);
		hit.result.z = hit.result.z || (resultMask & 4);
		hit.result.w = hit.result.w || (resultMask & 8);

		_mm_maskstore_ps(hit.tmin, _mm_castps_si128(hit_4), _mm_min_ps(hit.t_min, t_min));
#endif
	}

	hit.t_mini = _mm_and_si128(hit.t_mini, mask);
	hit.t_mini = _mm_or_si128(hit.t_mini, or_mask);

	if (hit.tmin[0] > hit.tmin[1])
		std::swap(hit.tmin[0], hit.tmin[1]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);
	if (hit.tmin[0] > hit.tmin[2])
		std::swap(hit.tmin[0], hit.tmin[2]);
	if (hit.tmin[1] > hit.tmin[3])
		std::swap(hit.tmin[1], hit.tmin[3]);
	if (hit.tmin[2] > hit.tmin[3])
		std::swap(hit.tmin[2], hit.tmin[3]);
	return hit;
}

void MBVHNode::merge_nodes(const BVHNode &node, const BVHNode *bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr)
{
	int numChildren = 0;
	get_bvh_node_info(node, bvhPool, numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (counts[idx] == -1) // not a leaf
		{
			const BVHNode &curNode = bvhPool[childs[idx]];
			const auto newIdx = poolPtr.fetch_add(1);
			MBVHNode &newNode = bvhTree[newIdx];
			childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx

			newNode.merge_nodes(curNode, bvhPool, bvhTree, poolPtr);
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		set_bounds(idx, vec3(1e34f), vec3(-1e34f));
		childs[idx] = 0;
		counts[idx] = 0;
	}
}

void MBVHNode::get_bvh_node_info(const BVHNode &node, const BVHNode *pool, int &numChildren)
{
	// Starting values
	childs = ivec4(-1);
	counts = ivec4(-1);
	numChildren = 0;

	if (node.is_leaf())
		throw std::runtime_error("Leaf nodes should not be attempted to be split");

	const BVHNode &orgLeftNode = pool[node.get_left_first()];
	const BVHNode &orgRightNode = pool[node.get_left_first() + 1];

	if (orgLeftNode.is_leaf()) // Node is a leaf
	{
		const int idx = numChildren++;
		set_bounds(idx, orgLeftNode.bounds);

		childs[idx] = orgLeftNode.get_left_first();
		counts[idx] = orgLeftNode.get_count();
	}
	else // Node has children
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = orgLeftNode.get_left_first();
		const int right = orgLeftNode.get_left_first() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		set_bounds(idx1, leftNode.bounds);
		set_bounds(idx2, rightNode.bounds);

		if (leftNode.is_leaf())
		{
			childs[idx1] = leftNode.get_left_first();
			counts[idx1] = leftNode.get_count();
		}
		else
		{
			childs[idx1] = left;
			counts[idx1] = -1;
		}

		if (rightNode.is_leaf())
		{
			childs[idx2] = rightNode.get_left_first();
			counts[idx2] = rightNode.get_count();
		}
		else
		{
			childs[idx2] = right;
			counts[idx2] = -1;
		}
	}

	if (orgRightNode.is_leaf())
	{
		// Node only has a single child
		const int idx = numChildren++;
		set_bounds(idx, orgRightNode.bounds);

		childs[idx] = orgRightNode.get_left_first();
		counts[idx] = orgRightNode.get_count();
	}
	else
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = orgRightNode.get_left_first();
		const int right = orgRightNode.get_left_first() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		set_bounds(idx1, leftNode.bounds);
		set_bounds(idx2, rightNode.bounds);

		if (leftNode.is_leaf())
		{
			childs[idx1] = leftNode.get_left_first();
			counts[idx1] = leftNode.get_count();
		}
		else
		{
			childs[idx1] = left;
			counts[idx1] = -1;
		}

		if (rightNode.is_leaf())
		{
			childs[idx2] = rightNode.get_left_first();
			counts[idx2] = rightNode.get_count();
		}
		else
		{
			childs[idx2] = right;
			counts[idx2] = -1;
		}
	}
}

void MBVHNode::sort_results(const float *tmin, int &a, int &b, int &c, int &d) const
{
	if (tmin[a] > tmin[b])
		std::swap(a, b);
	if (tmin[c] > tmin[d])
		std::swap(c, d);
	if (tmin[a] > tmin[c])
		std::swap(a, c);
	if (tmin[b] > tmin[d])
		std::swap(b, d);
	if (tmin[b] > tmin[c])
		std::swap(b, c);
}
bool MBVHNode::traverse_mbvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	bool valid = false;

	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count == 0)
			continue;

		if (count > 0)
		{
			// leaf node
			for (int i = 0; i < count; i++)
			{
				const uint primIdx = primIndices[leftFirst + i];
				const uvec3 &idx = indices[primIdx];

				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
				{
					valid = true;
					*hit_idx = primIdx;
				}
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}

	return valid;
}

bool MBVHNode::traverse_mbvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned int *primIndices, const glm::vec4 *vertices)
{
	bool valid = false;
	MBVHTraversal todo[32];
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
				const glm::uint primIdx = primIndices[leftFirst + i];
				const glm::uvec3 idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);

				if (rfw::triangle::intersect(org, dir, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
				{
					valid = true;
					*hit_idx = primIdx;
				}
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}

	return valid;
}

bool MBVHNode::traverse_mbvh(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
							 const unsigned *primIndices, const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	bool valid = false;
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count == 0)
			continue;

		if (count > -1) // leaf node
		{
			for (int i = 0; i < count; i++)
			{
				const glm::uint primIdx = primIndices[leftFirst + i];

				if (rfw::triangle::intersect_opt(org, dir, t_min, t, p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
				{
					valid = true;
					*hit_idx = primIdx;
				}
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx])
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}

	return valid;
}

int MBVHNode::traverse_mbvh(cpurt::RayPacket4 &packet, float t_min, const MBVHNode *nodes, const unsigned *primIndices, const glm::vec3 *p0s,
							const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask)
{
	MBVHTraversal todo[32];
	int stackptr = 0;
	int hitMask = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	*hit_mask = _mm_setzero_ps();

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count > -1) // leaf node
		{
			for (int i = 0; i < count; i++)
			{
#if 0 // Triangle intersection debugging
				int mask[4] = {0, 0, 0, 0};
				const auto primIdx = primIndices[leftFirst + i];
				for (int k = 0; k < 4; k++)
				{
					const vec3 org = vec3(packet.origin_x[k], packet.origin_y[k], packet.origin_z[k]);
					const vec3 dir = vec3(packet.direction_x[k], packet.direction_y[k], packet.direction_z[k]);

					if (triangle::intersect_opt(org, dir, t_min, &packet.t[k], p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
					{
						mask[k] = ~0;
						packet.primID[k] = primIdx;
					}
				}

				*hit_mask = _mm_or_ps(*hit_mask, _mm_castsi128_ps(_mm_setr_epi32(mask[0], mask[1], mask[2], mask[3])));
#else
				const auto primIdx = primIndices[leftFirst + i];
				__m128 store_mask = _mm_setzero_ps();
				hitMask |= triangle::intersect4(packet, p0s[primIdx], edge1s[primIdx], edge2s[primIdx], &store_mask);
				*hit_mask = _mm_or_ps(*hit_mask, store_mask);
				_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
#endif
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect4(packet, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}

	return hitMask;
}

int MBVHNode::traverse_mbvh(cpurt::RayPacket4 &packet, float t_min, const MBVHNode *nodes, const unsigned *primIndices, const glm::vec4 *vertices,
							const glm::uvec3 *indices, __m128 *hit_mask)
{
	MBVHTraversal todo[32];
	int stackptr = 0;
	int hitMask = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	*hit_mask = _mm_setzero_ps();

	while (stackptr >= 0)
	{
		const int leftFirst = todo[stackptr].leftFirst;
		const int count = todo[stackptr].count;
		stackptr--;

		if (count > -1) // leaf node
		{
			for (int i = 0; i < count; i++)
			{
				const auto primIdx = primIndices[leftFirst + i];
				const uvec3 idx = indices[primIdx];

#if 0 // Triangle intersection debugging
				int mask[4] = {0, 0, 0, 0};

				for (int k = 0; k < 4; k++)
				{
					const vec3 org = vec3(packet.origin_x[k], packet.origin_y[k], packet.origin_z[k]);
					const vec3 dir = vec3(packet.direction_x[k], packet.direction_y[k], packet.direction_z[k]);

					if (triangle::intersect(org, dir, t_min, &packet.t[k], vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					{
						mask[k] = ~0;
						packet.primID[k] = primIdx;
					}
				}

				*hit_mask = _mm_or_ps(*hit_mask, _mm_castsi128_ps(_mm_setr_epi32(mask[0], mask[1], mask[2], mask[3])));
#else
				__m128 store_mask = _mm_setzero_ps();
				hitMask |= triangle::intersect4(packet, vertices[idx.x], vertices[idx.y], vertices[idx.z], &store_mask);
				*hit_mask = _mm_or_ps(*hit_mask, store_mask);
				_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
#endif
			}
			continue;
		}

		const MBVHHit hit = nodes[leftFirst].intersect4(packet, t_min);
		for (int i = 3; i >= 0; i--)
		{ // reversed order, we want to check best nodes first
			const int idx = (hit.tmini[i] & 0b11);
			if (hit.result[idx] == 1)
			{
				stackptr++;
				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
				todo[stackptr].count = nodes[leftFirst].counts[idx];
			}
		}
	}

	return hitMask;
}

bool MBVHNode::traverse_mbvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
									const unsigned int *primIndices, const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		struct MBVHTraversal mTodo = todo[stackptr];
		stackptr--;

		if (mTodo.count > -1)
		{ // leaf node
			for (int i = 0; i < mTodo.count; i++)
			{
				const int primIdx = primIndices[mTodo.leftFirst + i];
				const uvec3 &idx = indices[primIdx];

				if (triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
			continue;
		}

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist, t_min);
		if (hit.result[0] || hit.result[1] || hit.result[2] || hit.result[3])
		{
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[mTodo.leftFirst].childs[idx];
					todo[stackptr].count = nodes[mTodo.leftFirst].counts[idx];
				}
			}
		}
	}

	// Nothing occluding
	return false;
}

bool MBVHNode::traverse_mbvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
									const unsigned int *primIndices, const glm::vec4 *vertices)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		struct MBVHTraversal mTodo = todo[stackptr];
		stackptr--;

		if (mTodo.count > -1) // leaf node
		{
			for (int i = 0; i < mTodo.count; i++)
			{
				const int primIdx = primIndices[mTodo.leftFirst + i];
				const uvec3 idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);

				if (triangle::intersect(org, dir, t_min, &maxDist, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
					return true;
			}
			continue;
		}

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist, t_min);
		if (hit.result[0] || hit.result[1] || hit.result[2] || hit.result[3])
		{
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[mTodo.leftFirst].childs[idx];
					todo[stackptr].count = nodes[mTodo.leftFirst].counts[idx];
				}
			}
		}
	}

	// Nothing occluding
	return false;
}

bool MBVHNode::traverse_mbvh_shadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes, const unsigned *primIndices,
									const glm::vec3 *p0s, const glm::vec3 *edge1s, const glm::vec3 *edge2s)
{
	MBVHTraversal todo[32];
	int stackptr = 0;

	todo[0].leftFirst = 0;
	todo[0].count = -1;

	const glm::vec3 dirInverse = 1.0f / dir;

	while (stackptr >= 0)
	{
		struct MBVHTraversal mTodo = todo[stackptr];
		stackptr--;

		if (mTodo.count > -1) // leaf node
		{
			for (int i = 0; i < mTodo.count; i++)
			{
				const int primIdx = primIndices[mTodo.leftFirst + i];
				const uvec3 idx = uvec3(primIdx * 3) + uvec3(0, 1, 2);

				if (rfw::triangle::intersect_opt(org, dir, t_min, &maxDist, p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
					return true;
			}
			continue;
		}

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist, t_min);
		if (hit.result[0] || hit.result[1] || hit.result[2] || hit.result[3])
		{
			for (int i = 3; i >= 0; i--)
			{ // reversed order, we want to check best nodes first
				const int idx = (hit.tmini[i] & 0b11);
				if (hit.result[idx] == 1)
				{
					stackptr++;
					todo[stackptr].leftFirst = nodes[mTodo.leftFirst].childs[idx];
					todo[stackptr].count = nodes[mTodo.leftFirst].counts[idx];
				}
			}
		}
	}

	// Nothing occluding
	return false;
}

void MBVHNode::validate(MBVHNode *nodes, unsigned maxPrimID, unsigned maxPoolPtr)
{
	for (int i = 0; i < 4; i++)
	{
		if (childs[i] < 0) // Unused nodes
			break;

		if (counts[i] >= 0)
		{
			for (int j = 0; j < counts[i]; j++)
			{
				if (childs[i] + j >= maxPrimID)
					throw std::runtime_error("Invalid node: PrimID is larger than maximum.");
			}
		}
		else
		{
			if (childs[i] >= maxPoolPtr)
				throw std::runtime_error("Invalid node: PoolPtr is larger than maximum.");
			nodes[childs[i]].validate(nodes, maxPrimID, maxPoolPtr);
		}
	}
}
