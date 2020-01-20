#include "../rfw.h"

#define USE_TOP_MBVH 1
#define USE_MBVH 1
#define REFIT 1

namespace rfw::bvh
{

void rfwMesh::set_geometry(const Mesh &mesh)
{
	triangles = mesh.triangles;
	vertices = mesh.vertices;
	if (mesh.hasIndices())
		indices = mesh.indices;
	else
		indices = nullptr;

	const bool rebuild = !bvh || (vertexCount != mesh.vertexCount);

	vertexCount = mesh.vertexCount;
	triangleCount = mesh.triangleCount;

#if REFIT
	if (rebuild) // Full rebuild of BVH
	{
		delete bvh;
		delete mbvh;

		if (mesh.hasIndices())
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
		else
			bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

		bvh->construct_bvh();
		mbvh = new MBVHTree(bvh);
		mbvh->construct_bvh();
	}
	else // Keep same BVH but refit nodes
	{
		if (mesh.hasIndices())
			mbvh->refit(mesh.vertices, mesh.indices);
		else
			mbvh->refit(mesh.vertices);
	}
#else
	delete bvh;
	delete mbvh;

	if (mesh.hasIndices())
		bvh = new BVHTree(mesh.vertices, mesh.vertexCount, mesh.indices, mesh.triangleCount);
	else
		bvh = new BVHTree(mesh.vertices, mesh.vertexCount);

	bvh->construct_bvh();
	mbvh = new MBVHTree(bvh);
	mbvh->construct_bvh();
#endif
}

void TopLevelBVH::construct_bvh()
{
	bvh_nodes.clear();
	AABB rootBounds = {};
	for (const auto &aabb : instance_aabbs)
		rootBounds.grow(aabb);

	bvh_nodes.resize(aabbs.size() * 2);
	prim_indices.resize(aabbs.size());
	for (uint i = 0, s = static_cast<uint>(prim_indices.size()); i < s; i++)
		prim_indices[i] = i;

	pool_ptr.store(2);
	bvh_nodes[0].bounds = rootBounds;
	bvh_nodes[0].bounds.leftFirst = 0;
	bvh_nodes[0].bounds.count = static_cast<int>(instance_aabbs.size());

	// Use less split bins to speed up construction
	// allow only 1 primitive per node as intersecting nodes is cheaper than intersecting bottom level BVHs
	bvh_nodes[0].subdivide<7, 32, 1>(instance_aabbs.data(), bvh_nodes.data(), prim_indices.data(), 1, pool_ptr);
	if (pool_ptr > 2)
	{
		bvh_nodes[0].bounds.count = -1;
		bvh_nodes[0].set_left_first(2);
	}
	else
	{
		bvh_nodes[0].bounds.count = static_cast<int>(instance_aabbs.size());
		bvh_nodes[0].set_left_first(0);
	}
	bvh_nodes.resize(bvh_nodes.size());

	mbvh_nodes.resize(bvh_nodes.size()); // We'll store at most the original nodes in terms of size
	if (pool_ptr <= 4)
	{
		int num_children = 0;
		mbvh_nodes[0].merge_node(bvh_nodes[0], bvh_nodes, num_children);
	}
	else
	{
		mpool_ptr.store(1);
		mbvh_nodes[0].merge_nodes(bvh_nodes[0], bvh_nodes, mbvh_nodes.data(), mpool_ptr);
	}

	mbvh_nodes.resize(mbvh_nodes.size());
}

void TopLevelBVH::refit()
{
	if (count_changed)
	{
		bvh_nodes.clear();
		AABB rootBounds = {};
		for (const auto &aabb : instance_aabbs)
			rootBounds.grow(aabb);

		bvh_nodes.resize(aabbs.size() * 2);
		prim_indices.resize(aabbs.size());
		for (uint i = 0, s = static_cast<uint>(prim_indices.size()); i < s; i++)
			prim_indices[i] = i;

		pool_ptr.store(2);
		bvh_nodes[0].bounds = rootBounds;
		bvh_nodes[0].bounds.leftFirst = 0;
		bvh_nodes[0].bounds.count = static_cast<int>(instance_aabbs.size());

		// Use less split bins to speed up construction
		// allow only 1 primitive per node as intersecting nodes is cheaper than intersecting bottom level BVHs
		bvh_nodes[0].subdivide<7, 32, 1>(instance_aabbs.data(), bvh_nodes.data(), prim_indices.data(), 1, pool_ptr);
		if (pool_ptr > 2)
		{
			bvh_nodes[0].bounds.count = -1;
			bvh_nodes[0].set_left_first(2);
		}
		else
		{
			bvh_nodes[0].bounds.count = static_cast<int>(instance_aabbs.size());
			bvh_nodes[0].set_left_first(0);
		}
	}
	else
	{
		AABB rootBounds = {};
		for (const auto &aabb : instance_aabbs)
			rootBounds.grow(aabb);

		bvh_nodes[0].refit(bvh_nodes.data(), prim_indices.data(), instance_aabbs.data());
	}

	if (pool_ptr <= 4) // Original tree first in single MBVH node
	{
		mbvh_nodes.resize(1);
		MBVHNode &mRootNode = mbvh_nodes[0];
		const AABB invalidAABB = AABB();

		for (int i = 0, s = pool_ptr; i < s; i++)
		{
			BVHNode &curNode = bvh_nodes[i];

			if (curNode.is_leaf())
			{
				mRootNode.childs[i] = curNode.get_left_first();
				mRootNode.counts[i] = curNode.get_count();
				mRootNode.set_bounds(i, curNode.bounds);
			}
			else
			{
				mRootNode.childs[i] = 0;
				mRootNode.counts[i] = 0;
				mRootNode.set_bounds(i, invalidAABB);
			}
		}

		for (int i = pool_ptr; i < 4; i++)
		{
			mRootNode.childs[i] = 0;
			mRootNode.counts[i] = 0;
			mRootNode.set_bounds(i, invalidAABB);
		}
	}
	else
	{
		mpool_ptr.store(1);
		mbvh_nodes.resize(bvh_nodes.size()); // We'll store at most the original nodes in terms of size
		mbvh_nodes[0].merge_nodes(bvh_nodes[0], bvh_nodes, mbvh_nodes.data(), mpool_ptr);
	}
}

void TopLevelBVH::set_instance(size_t idx, glm::mat4 transform, rfwMesh *tree, AABB boundingBox)
{
	while (idx >= static_cast<int>(instance_meshes.size()))
	{
		simd::matrix4 m = glm::mat4(1.0f);

		count_changed = true;
		instance_aabbs.push_back(boundingBox);
		aabbs.push_back(boundingBox);
		instance_meshes.push_back(tree);
		matrices.push_back(m);
		normal_matrices.push_back(m);
		inverse_matrices.push_back(m);
	}

	aabbs[idx] = boundingBox;
	instance_meshes[idx] = tree;
	matrices[idx] = transform;
	inverse_matrices[idx] = inverse(transform);
	normal_matrices[idx] = mat4(transpose(inverse(mat3(transform))));
	instance_aabbs[idx] = calculate_world_bounds(boundingBox, matrices[idx]);
}

AABB TopLevelBVH::calculate_world_bounds(const AABB &originalBounds, const simd::matrix4 &matrix)
{
	const __m128 p1 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f));
	const __m128 p5 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f));

	const __m128 p2 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f));
	const __m128 p3 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f));

	const __m128 p4 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f));
	const __m128 p6 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f));

	const __m128 p7 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f));
	const __m128 p8 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f));

	AABB transformedAABB = {};
	transformedAABB.grow(p1);
	transformedAABB.grow(p2);
	transformedAABB.grow(p3);
	transformedAABB.grow(p4);
	transformedAABB.grow(p5);
	transformedAABB.grow(p6);
	transformedAABB.grow(p7);
	transformedAABB.grow(p8);

	const __m128 epsilon4 = _mm_set1_ps(1e-5f);
	transformedAABB.bmin4 = _mm_sub_ps(transformedAABB.bmin4, epsilon4);
	transformedAABB.bmax4 = _mm_sub_ps(transformedAABB.bmax4, epsilon4);

	return transformedAABB;
}

const rfw::Triangle &TopLevelBVH::get_triangle(int instID, int primID) const
{
	assert(instID > 0 && instID < instance_meshes.size());
	return instance_meshes[instID]->triangles[primID];
}

const rfw::simd::matrix4 &TopLevelBVH::get_normal_matrix(int instID) const { return normal_matrices[instID]; }

const rfw::simd::matrix4 &TopLevelBVH::get_instance_matrix(int instID) const { return matrices[instID]; }

} // namespace rfw::bvh