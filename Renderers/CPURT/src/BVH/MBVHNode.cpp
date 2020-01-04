#define GLM_FORCE_AVX
#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"

#include <thread>
#include <future>

using namespace glm;
using namespace rfw;

void MBVHNode::SetBounds(unsigned int nodeIdx, const vec3 &min, const vec3 &max)
{
	this->bminx[nodeIdx] = min.x;
	this->bminy[nodeIdx] = min.y;
	this->bminz[nodeIdx] = min.z;

	this->bmaxx[nodeIdx] = max.x;
	this->bmaxy[nodeIdx] = max.y;
	this->bmaxz[nodeIdx] = max.z;
}

void MBVHNode::SetBounds(unsigned int nodeIdx, const AABB &bounds)
{
	this->bminx[nodeIdx] = bounds.xMin;
	this->bminy[nodeIdx] = bounds.yMin;
	this->bminz[nodeIdx] = bounds.zMin;

	this->bmaxx[nodeIdx] = bounds.xMax;
	this->bmaxy[nodeIdx] = bounds.yMax;
	this->bmaxz[nodeIdx] = bounds.zMax;
}

MBVHHit MBVHNode::intersect(const glm::vec3 &org, const glm::vec3 &dirInverse, float *t) const
{
#if 1
	static const __m128i mask = _mm_set1_epi32(0xFFFFFFFCu);
	static const __m128i or_mask = _mm_set_epi32(0b11, 0b10, 0b01, 0b00);

	MBVHHit hit{};

	__m128 orgComponent = _mm_set1_ps(org.x);
	__m128 dirComponent = _mm_set1_ps(dirInverse.x);
	__m256 orgComponent8 = _mm256_set_m128(orgComponent, orgComponent);
	__m256 dirComponent8 = _mm256_set_m128(dirComponent, dirComponent);

#if 0
	union {
		struct
		{
			__m128 t1, t2;
		};
		__m256 t8;
	};

	t8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set_m128(bmaxx_4, bminx_4), orgComponent8), dirComponent8);
#else
	__m128 t1 = _mm_mul_ps(_mm_sub_ps(bminx_4, orgComponent), dirComponent);
	__m128 t2 = _mm_mul_ps(_mm_sub_ps(bmaxx_4, orgComponent), dirComponent);
#endif

	hit.t_min = _mm_min_ps(t1, t2);
	__m128 t_max = _mm_max_ps(t1, t2);

	orgComponent = _mm_set1_ps(org.y);
	dirComponent = _mm_set1_ps(dirInverse.y);

#if 0
	orgComponent8 = _mm256_set_m128(orgComponent, orgComponent);
	dirComponent8 = _mm256_set_m128(dirComponent, dirComponent);
	t8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set_m128(bmaxy_4, bminy_4), orgComponent8), dirComponent8);
#else
	t1 = _mm_mul_ps(_mm_sub_ps(bminy_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxy_4, orgComponent), dirComponent);
#endif

	hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	orgComponent = _mm_set1_ps(org.z);
	dirComponent = _mm_set1_ps(dirInverse.z);

#if 0
	orgComponent8 = _mm256_set_m128(orgComponent, orgComponent);
	dirComponent8 = _mm256_set_m128(dirComponent, dirComponent);
	t8 = _mm256_mul_ps(_mm256_sub_ps(_mm256_set_m128(bmaxz_4, bminz_4), orgComponent8), dirComponent8);
#else
	t1 = _mm_mul_ps(_mm_sub_ps(bminz_4, orgComponent), dirComponent);
	t2 = _mm_mul_ps(_mm_sub_ps(bmaxz_4, orgComponent), dirComponent);
#endif

	hit.t_min = _mm_max_ps(hit.t_min, _mm_min_ps(t1, t2));
	t_max = _mm_min_ps(t_max, _mm_max_ps(t1, t2));

	hit.t_mini = _mm_and_si128(hit.t_mini, mask);
	hit.t_mini = _mm_or_si128(hit.t_mini, or_mask);
	const __m128 greaterThan0 = _mm_cmpgt_ps(t_max, _mm_set1_ps(0.0f));
	const __m128 lessThanEqualMax = _mm_cmple_ps(hit.t_min, t_max);
	const __m128 lessThanT = _mm_cmplt_ps(hit.t_min, _mm_set1_ps(*t));

	const __m128 result = _mm_and_ps(greaterThan0, _mm_and_ps(lessThanEqualMax, lessThanT));
	const int resultMask = _mm_movemask_ps(result);
	hit.result = glm::bvec4(resultMask & 1, resultMask & 2, resultMask & 4, resultMask & 8);

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

	hit.result = greaterThan(tmax, glm::vec4(0.0f)) && lessThanEqual(hit.tmin4, tmax) && lessThan(hit.tmin4, glm::vec4(*t));

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

void MBVHNode::MergeNodes(const BVHNode &node, const BVHNode *bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (this->counts[idx] == -1)
		{ // not a leaf
			const BVHNode &curNode = bvhPool[this->childs[idx]];
			if (curNode.IsLeaf())
			{
				this->counts[idx] = curNode.GetCount();
				this->childs[idx] = curNode.GetLeftFirst();
				this->SetBounds(idx, curNode.bounds);
			}
			else
			{
				const auto newIdx = poolPtr.fetch_add(1);
				MBVHNode &newNode = bvhTree[newIdx];
				this->childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
				this->counts[idx] = -1;
				this->SetBounds(idx, curNode.bounds);
				newNode.MergeNodes(curNode, bvhPool, bvhTree, poolPtr);
			}
		}
	}

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		this->SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		this->counts[idx] = 0;
	}
}

void MBVHNode::MergeNodesMT(const BVHNode &node, const BVHNode *bvhPool, MBVHNode *bvhTree, std::atomic_int &poolPtr, std::atomic_int &threadCount, bool thread)
{
	int numChildren;
	GetBVHNodeInfo(node, bvhPool, numChildren);

	std::vector<std::future<void>> threads;

	// invalidate any remaining children
	for (int idx = numChildren; idx < 4; idx++)
	{
		SetBounds(idx, vec3(1e34f), vec3(-1e34f));
		counts[idx] = 0;
	}

	for (int idx = 0; idx < numChildren; idx++)
	{
		if (counts[idx] < 0) // not a leaf
		{
			const BVHNode *curNode = &bvhPool[childs[idx]];

			if (counts[idx] >= 0) // Node is leaf node, no need to subdivide
				continue;

			const auto newIdx = poolPtr.fetch_add(1);
			MBVHNode *newNode = &bvhTree[newIdx];
			childs[idx] = newIdx; // replace BVHNode idx with MBVHNode idx
			SetBounds(idx, curNode->bounds);
			newNode->MergeNodesMT(*curNode, bvhPool, bvhTree, poolPtr, threadCount, !thread);
		}
	}

	for (auto &t : threads)
		t.get();
}

void MBVHNode::GetBVHNodeInfo(const BVHNode &node, const BVHNode *pool, int &numChildren)
{
	// Starting values
	childs = ivec4(-1);
	counts = ivec4(-1);
	numChildren = 0;

	if (node.IsLeaf())
	{
		throw std::runtime_error("This node shouldn't be a leaf");
		return;
	}

	const BVHNode &orgLeftNode = pool[node.GetLeftFirst()];
	const BVHNode &orgRightNode = pool[node.GetLeftFirst() + 1];

	if (orgLeftNode.IsLeaf()) // Node is a leaf
	{
		const int idx = numChildren++;
		SetBounds(idx, orgLeftNode.bounds);

		childs[idx] = orgLeftNode.GetLeftFirst();
		counts[idx] = orgLeftNode.GetCount();
	}
	else // Node has children
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = orgLeftNode.GetLeftFirst();
		const int right = orgLeftNode.GetLeftFirst() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		SetBounds(idx1, leftNode.bounds);
		SetBounds(idx2, rightNode.bounds);

		if (leftNode.IsLeaf())
			childs[idx1] = leftNode.GetLeftFirst();
		else
			childs[idx1] = left;

		if (rightNode.IsLeaf())
			childs[idx2] = rightNode.GetLeftFirst();
		else
			childs[idx2] = right;

		counts[idx1] = leftNode.GetCount();
		counts[idx2] = rightNode.GetCount();
	}

	if (orgRightNode.IsLeaf())
	{
		// Node only has a single child
		const int idx = numChildren++;
		SetBounds(idx, orgRightNode.bounds);

		childs[idx] = orgRightNode.GetLeftFirst();
		counts[idx] = orgRightNode.GetCount();
	}
	else
	{
		const int idx1 = numChildren++;
		const int idx2 = numChildren++;

		const int left = orgRightNode.GetLeftFirst();
		const int right = orgRightNode.GetLeftFirst() + 1;

		const BVHNode &leftNode = pool[left];
		const BVHNode &rightNode = pool[right];

		SetBounds(idx1, leftNode.bounds);
		SetBounds(idx2, rightNode.bounds);

		if (leftNode.IsLeaf())
			childs[idx1] = leftNode.GetLeftFirst();
		else
			childs[idx1] = left;

		if (rightNode.IsLeaf())
			childs[idx2] = rightNode.GetLeftFirst();
		else
			childs[idx2] = right;

		counts[idx1] = leftNode.GetCount();
		counts[idx2] = rightNode.GetCount();
	}
}

void MBVHNode::SortResults(const float *tmin, int &a, int &b, int &c, int &d) const
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
bool MBVHNode::traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
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

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t);
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

bool MBVHNode::traverseMBVH(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float *t, int *hit_idx, const MBVHNode *nodes,
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

		if (count > -1)
		{ // leaf node
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

		const MBVHHit hit = nodes[leftFirst].intersect(org, dirInverse, t);
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

bool MBVHNode::traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
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

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist);
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

bool MBVHNode::traverseMBVHShadow(const glm::vec3 &org, const glm::vec3 &dir, float t_min, float maxDist, const MBVHNode *nodes,
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

		const MBVHHit hit = nodes[mTodo.leftFirst].intersect(org, dirInverse, &maxDist);
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
