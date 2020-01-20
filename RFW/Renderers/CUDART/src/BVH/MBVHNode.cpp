#include "../PCH.h"

using namespace glm;

namespace rfw::bvh
{

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
	MBVHHit hit;

#if 1
	static const __m128i mask = _mm_set1_epi32(0xFFFFFFFCu);
	static const __m128i or_mask = _mm_set_epi32(0b11, 0b10, 0b01, 0b00);

	__m128 orgComponent = _mm_set1_ps(org.x);
	__m128 dirComponent = _mm_set1_ps(dirInverse.x);

	__m128 t1 = _mm_mul_ps(_mm_sub_ps(bminx_4, orgComponent), dirComponent);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(bmaxx_4, orgComponent), dirComponent);

	hit.tmin4 = _mm_min_ps(t1, t2);
	__m128 t_max = _mm_max_ps(t1, t2);

	orgComponent = _mm_set1_ps(org.y);
	dirComponent = _mm_set1_ps(dirInverse.y);

	t1 = _mm_mul_ps(_mm_sub_ps(bminy_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxy_4, orgComponent), dirComponent);

	hit.tmin4 = _mm_max_ps(hit.tmin4, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	orgComponent = _mm_set1_ps(org.z);
	dirComponent = _mm_set1_ps(dirInverse.z);

	t1 = _mm_mul_ps(_mm_sub_ps(bminz_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxz_4, orgComponent), dirComponent);

	hit.tmin4 = _mm_max_ps(hit.tmin4, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	const __m128 greaterThan0 = _mm_cmpge_ps(t_max, _mm_set1_ps(t_min));
	const __m128 lessThanEqualMax = _mm_cmplt_ps(hit.tmin4, t_max);
	const __m128 lessThanT = _mm_cmplt_ps(hit.tmin4, _mm_set1_ps(*t));

	const __m128 result = _mm_and_ps(greaterThan0, _mm_and_ps(lessThanEqualMax, lessThanT));
	const int resultMask = _mm_movemask_ps(result);

	hit.tmini4 = _mm_and_si128(hit.tmini4, mask);
	hit.tmini4 = _mm_or_si128(hit.tmini4, or_mask);
	hit.result = bvec4(resultMask & 1, resultMask & 2, resultMask & 4, resultMask & 8);
#else
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
#endif

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

void MBVHNode::merge_nodes(const BVHNode &node, const rfw::utils::ArrayProxy<BVHNode> bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr)
{
	if (node.is_leaf())
		throw std::runtime_error("Leaf nodes should not be attempted to be split");

	const int left = node.get_left_first();
	const int right = left + 1;

	const BVHNode &left_node = bvhPool[left];
	const BVHNode &right_node = bvhPool[right];

	int numChildren = 0;

	merge_node(node, bvhPool, numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (childs[idx] < 0) // Invalidate invalid children
		{
			set_bounds(idx, vec3(1e34f), vec3(-1e34f));
			childs[idx] = 0;
			counts[idx] = 0;
			continue;
		}

		if (counts[idx] < 0) // not a leaf
		{
			const BVHNode &curNode = bvhPool[childs[idx]];
			const auto newIdx = poolPtr.fetch_add(1);
			MBVHNode &newNode = bvhTree[newIdx];
			childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx

			newNode.merge_nodes(curNode, bvhPool, bvhTree, poolPtr);
		}
	}
}

void MBVHNode::merge_node(const BVHNode &node, const rfw::utils::ArrayProxy<BVHNode> pool, int &numChildren)
{
	// Starting values
	childs = ivec4(-1);
	counts = ivec4(-1);
	numChildren = 0;

	const BVHNode &left_node = pool[node.get_left_first()];
	const BVHNode &right_node = pool[node.get_left_first() + 1];

	if (left_node.is_leaf()) // Node is a leaf
	{
		const int idx = numChildren++;
		childs[idx] = left_node.get_left_first();
		counts[idx] = left_node.get_count();
		set_bounds(idx, left_node.bounds);
	}
	else // Node has children
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = left_node.get_left_first();
		const int right = left_node.get_left_first() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		set_bounds(idx1, leftNode.bounds);
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

		set_bounds(idx2, rightNode.bounds);
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

	if (right_node.is_leaf())
	{
		// Node only has a single child
		const int idx = numChildren++;
		set_bounds(idx, right_node.bounds);

		childs[idx] = right_node.get_left_first();
		counts[idx] = right_node.get_count();
	}
	else
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = right_node.get_left_first();
		const int right = right_node.get_left_first() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		set_bounds(idx1, leftNode.bounds);
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

		set_bounds(idx2, rightNode.bounds);
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

	// In case this quad node isn't filled & not all nodes are leaf nodes, merge 1 more node
	if (numChildren == 3)
	{
		for (int i = 0; i < 3; i++)
		{
			if (counts[i] >= 0)
				continue;

			const int left = childs[i];
			const int right = left + 1;

			const BVHNode &left_sub_node = pool[left];
			const BVHNode &right_sub_node = pool[right];

			// Overwrite current node
			set_bounds(i, left_sub_node.bounds);
			if (left_sub_node.is_leaf())
			{
				counts[i] = left_sub_node.get_count();
				childs[i] = left_sub_node.get_left_first();
			}
			else
			{
				counts[i] = -1;
				childs[i] = left;
			}

			// Add its right node
			set_bounds(numChildren, right_sub_node.bounds);
			if (right_sub_node.is_leaf())
			{
				counts[numChildren] = right_sub_node.get_count();
				childs[numChildren] = right_sub_node.get_left_first();
			}
			else
			{
				counts[numChildren] = -1;
				childs[numChildren] = right;
			}

			numChildren += 1;
			break;
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

void MBVHNode::validate(const rfw::utils::ArrayProxy<MBVHNode> nodes, const rfw::utils::ArrayProxy<uint> primIDs, uint maxPoolPtr,
						uint maxPrimIndex) const
{
	for (int i = 0; i < 4; i++)
	{
		if (childs[i] < 0) // Unused nodes
			continue;

		if (counts[i] >= 0)
		{
			for (int j = 0; j < counts[i]; j++)
			{
				const uint prim_index = childs[i] + j;
				if (prim_index >= primIDs.size())
					throw std::runtime_error("Invalid node: PrimID is larger than maximum.");
				else if (primIDs[prim_index] >= maxPrimIndex)
					throw std::runtime_error("Invalid node: PrimID points to object larger than maximum");
			}
		}
		else
		{
			if (childs[i] >= maxPoolPtr)
				throw std::runtime_error("Invalid node: PoolPtr is larger than maximum.");
			nodes[childs[i]].validate(nodes, primIDs, maxPoolPtr, maxPrimIndex);
		}
	}
}

} // namespace rfw::bvh