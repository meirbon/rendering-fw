#include "../PCH.h"

#define USE_TOP_MBVH 1
#define USE_MBVH 1
#define REFIT 1

namespace rfw::bvh
{

void BVHMesh::set_geometry(const Mesh &mesh)
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
	m_Nodes.clear();
	AABB rootBounds = {};
	for (const auto &aabb : transformedAABBs)
		rootBounds.grow(aabb);

	m_Nodes.resize(boundingBoxes.size() * 2);
	m_PrimIndices.resize(boundingBoxes.size());
	for (uint i = 0, s = static_cast<uint>(m_PrimIndices.size()); i < s; i++)
		m_PrimIndices[i] = i;

	m_PoolPtr.store(2);
	m_Nodes[0].bounds = rootBounds;
	m_Nodes[0].bounds.leftFirst = 0;
	m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());

	// Use less split bins to speed up construction
	// allow only 1 primitive per node as intersecting nodes is cheaper than intersecting bottom level BVHs
	m_Nodes[0].subdivide<7, 32, 1>(transformedAABBs.data(), m_Nodes.data(), m_PrimIndices.data(), 1, m_PoolPtr);
	if (m_PoolPtr > 2)
	{
		m_Nodes[0].bounds.count = -1;
		m_Nodes[0].set_left_first(2);
	}
	else
	{
		m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());
		m_Nodes[0].set_left_first(0);
	}
	m_Nodes.resize(m_Nodes.size());

	m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
	if (m_PoolPtr <= 4)
	{
		int num_children = 0;
		m_MNodes[0].merge_node(m_Nodes[0], m_Nodes, num_children);
	}
	else
	{
		m_MPoolPtr.store(1);
		m_MNodes[0].merge_nodes(m_Nodes[0], m_Nodes, m_MNodes.data(), m_MPoolPtr);
	}

	m_MNodes.resize(m_MNodes.size());
}

void TopLevelBVH::refit()
{
	if (instanceCountChanged)
	{
		m_Nodes.clear();
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.grow(aabb);

		m_Nodes.resize(boundingBoxes.size() * 2);
		m_PrimIndices.resize(boundingBoxes.size());
		for (uint i = 0, s = static_cast<uint>(m_PrimIndices.size()); i < s; i++)
			m_PrimIndices[i] = i;

		m_PoolPtr.store(2);
		m_Nodes[0].bounds = rootBounds;
		m_Nodes[0].bounds.leftFirst = 0;
		m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());

		// Use less split bins to speed up construction
		// allow only 1 primitive per node as intersecting nodes is cheaper than intersecting bottom level BVHs
		m_Nodes[0].subdivide<7, 32, 1>(transformedAABBs.data(), m_Nodes.data(), m_PrimIndices.data(), 1, m_PoolPtr);
		if (m_PoolPtr > 2)
		{
			m_Nodes[0].bounds.count = -1;
			m_Nodes[0].set_left_first(2);
		}
		else
		{
			m_Nodes[0].bounds.count = static_cast<int>(transformedAABBs.size());
			m_Nodes[0].set_left_first(0);
		}
	}
	else
	{
		AABB rootBounds = {};
		for (const auto &aabb : transformedAABBs)
			rootBounds.grow(aabb);

		m_Nodes[0].refit(m_Nodes.data(), m_PrimIndices.data(), transformedAABBs.data());
	}

	if (m_PoolPtr <= 4) // Original tree first in single MBVH node
	{
		m_MNodes.resize(1);
		MBVHNode &mRootNode = m_MNodes[0];
		const AABB invalidAABB = AABB();

		for (int i = 0, s = m_PoolPtr; i < s; i++)
		{
			BVHNode &curNode = m_Nodes[i];

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

		for (int i = m_PoolPtr; i < 4; i++)
		{
			mRootNode.childs[i] = 0;
			mRootNode.counts[i] = 0;
			mRootNode.set_bounds(i, invalidAABB);
		}
	}
	else
	{
		m_MPoolPtr.store(1);
		m_MNodes.resize(m_Nodes.size()); // We'll store at most the original nodes in terms of size
		m_MNodes[0].merge_nodes(m_Nodes[0], m_Nodes, m_MNodes.data(), m_MPoolPtr);
	}
}

void TopLevelBVH::set_instance(int idx, glm::mat4 transform, BVHMesh *tree, AABB boundingBox)
{
	while (idx >= static_cast<int>(accelerationStructures.size()))
	{
		instanceCountChanged = true;
		transformedAABBs.emplace_back();
		boundingBoxes.emplace_back();
		accelerationStructures.emplace_back();
		instanceMatrices.emplace_back();
		inverseNormalMatrices.emplace_back();
		inverseMatrices.emplace_back();
	}

	boundingBoxes[idx] = boundingBox;
	accelerationStructures[idx] = tree;
	instanceMatrices[idx] = transform;
	inverseMatrices[idx] = inverse(transform);
	inverseNormalMatrices[idx] = mat4(transpose(inverse(mat3(transform))));
	transformedAABBs[idx] = calculate_world_bounds(boundingBox, instanceMatrices[idx]);
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
	assert(instID > 0 && instID < accelerationStructures.size());
	return accelerationStructures[instID]->triangles[primID];
}

const rfw::simd::matrix4 &TopLevelBVH::get_normal_matrix(int instID) const { return inverseNormalMatrices[instID]; }

const rfw::simd::matrix4 &TopLevelBVH::get_instance_matrix(int instID) const { return instanceMatrices[instID]; }

} // namespace rfw::bvh