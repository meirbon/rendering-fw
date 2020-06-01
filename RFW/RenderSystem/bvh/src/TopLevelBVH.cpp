#include <bvh/BVH.h>

#include <utils/Logger.h>
#include <utils/Timer.h>

#include <tbb/parallel_for.h>

#define USE_TOP_MBVH 1
#define USE_MBVH 1
#define TOP_PACKET_MBVH 1
#define PACKET_MBVH 1
#define REFIT 1

namespace rfw::bvh
{

void rfwMesh::set_geometry(const rfw::Mesh &mesh)
{
	triangles = mesh.triangles;
	vertices = mesh.vertices;
	if (mesh.hasIndices())
		indices = mesh.indices;
	else
		indices = nullptr;

	const bool rebuild = !bvh || (vertexCount != mesh.vertexCount);

	vertexCount = static_cast<int>(mesh.vertexCount);
	triangleCount = static_cast<int>(mesh.triangleCount);

	if (rebuild) // Full rebuild of BVH
	{
		const int vCount = static_cast<int>(mesh.vertexCount);
		const int tCount = static_cast<int>(mesh.triangleCount);

		if (mesh.hasIndices())
			bvh = std::make_unique<BVHTree>(mesh.vertices, vCount, mesh.indices, tCount);
		else
			bvh = std::make_unique<BVHTree>(mesh.vertices, vCount);

		bvh->construct(BVHTree::SpatialSAH);

		mbvh = std::make_unique<MBVHTree>(bvh.get());
		mbvh->construct();
	}
	else // Keep same BVH but refit nodes
	{
		if (mesh.hasIndices())
			mbvh->refit(mesh.vertices, mesh.indices);
		else
			mbvh->refit(mesh.vertices);
	}
}

void TopLevelBVH::construct_bvh()
{
	if (bvh.has_value())
	{
		rtbvh::free_bvh(bvh.value());
		bvh = std::nullopt;
	}

	bvh = rtbvh::create_bvh(reinterpret_cast<const rtbvh::RTAABB *>(instance_aabbs.data()), instance_aabbs.size(),
							reinterpret_cast<const float *>(centers.data()), sizeof(glm::vec4),
							rtbvh::BVHType::BinnedSAH);

	if (mbvh.has_value())
	{
		rtbvh::free_mbvh(mbvh.value());
		mbvh = std::nullopt;
	}

	mbvh = rtbvh::create_mbvh(bvh.value());
}

void TopLevelBVH::refit()
{
	if (count_changed)
	{
		if (bvh.has_value())
		{
			rtbvh::free_bvh(bvh.value());
			bvh = std::nullopt;
		}

		bvh = rtbvh::create_bvh(reinterpret_cast<const rtbvh::RTAABB *>(instance_aabbs.data()), instance_aabbs.size(),
								reinterpret_cast<const float *>(centers.data()), sizeof(glm::vec4),
								rtbvh::BVHType::BinnedSAH);
	}
	else
	{
		rtbvh::refit(reinterpret_cast<const rtbvh::RTAABB *>(instance_aabbs.data()), bvh.value());
	}

	if (mbvh.has_value())
	{
		rtbvh::free_mbvh(mbvh.value());
		mbvh = std::nullopt;
	}

	mbvh = rtbvh::create_mbvh(bvh.value());
}

const rfw::Triangle *TopLevelBVH::intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID,
											int *instID, glm::vec2 *bary, float t_min) const
{
	const simd::vector4 org = vec4(origin, 1.0f);
	const simd::vector4 dir = vec4(direction, 0.0f);

#if USE_TOP_MBVH
	if (MBVHNode::traverse_mbvh(origin, direction, t_min, t, instID,
								reinterpret_cast<const MBVHNode *>(mbvh.value().nodes), mbvh.value().indices,
								[&](const int instance) {
#else
	if (BVHNode::traverse_bvh(origin, direction, t_min, t, instID, reinterpret_cast<const BVHNode *>(bvh.value().nodes),
							  bvh.value().indices, [&](const int instance) {
#endif
									const simd::vector4 new_origin = inverse_matrices[instance] * org;
									const simd::vector4 new_direction = inverse_matrices[instance] * dir;

									const glm::vec3 org = new_origin.vec;
									const glm::vec3 dir = new_direction.vec;

#if USE_MBVH
									return instance_meshes[instance]->mbvh->traverse(org, dir, t_min, t, primID, bary);
#else
								  return instance_meshes[instance]->bvh->traverse(org, dir, t_min, t, primID, bary);
#endif
								}))
	{
		return &instance_meshes[*instID]->triangles[*primID];
	}

	return nullptr;
}

const rfw::Triangle *TopLevelBVH::intersect(const vec3 &origin, const vec3 &direction, float *t, int *primID,
											int *instID, float t_min) const
{
	const simd::vector4 org = vec4(origin, 1.0f);
	const simd::vector4 dir = vec4(direction, 0.0f);

#if USE_TOP_MBVH
	if (MBVHNode::traverse_mbvh(origin, direction, t_min, t, instID,
								reinterpret_cast<const MBVHNode *>(mbvh.value().nodes), mbvh.value().indices,
								[&](const int instance) {
#else
	if (BVHNode::traverse_bvh(origin, direction, t_min, t, instID, reinterpret_cast<const BVHNode *>(bvh.value().nodes),
							  bvh.value().indices, [&](const int instance) {
#endif
									const simd::vector4 new_origin = inverse_matrices[instance] * org;
									const simd::vector4 new_direction = inverse_matrices[instance] * dir;

									const glm::vec3 org = new_origin.vec;
									const glm::vec3 dir = new_direction.vec;

#if USE_MBVH
									return instance_meshes[instance]->mbvh->traverse(org, dir, t_min, t, primID);
#else
								  return instance_meshes[instance]->bvh->traverse(org, dir, t_min, t, primID);
#endif
								}))
	{
		return &instance_meshes[*instID]->triangles[*primID];
	}

	return nullptr;
}

bool TopLevelBVH::is_occluded(const vec3 &origin, const vec3 &direction, float t_max, float t_min) const
{

#if USE_TOP_MBVH
	return MBVHNode::traverse_mbvh_shadow(
		origin, direction, t_min, t_max, reinterpret_cast<const MBVHNode *>(mbvh.value().nodes), mbvh.value().indices,
		[&](const int instance) {
#else
	return BVHNode::traverse_bvh_shadow(
		origin, direction, t_min, t_max, reinterpret_cast<const BVHNode *>(bvh.value().nodes), bvh.value().indices,
		[&](const int instance) {
#endif
			const vec3 new_origin = inverse_matrices[instance] * vec4(origin, 1);
			const vec3 new_direction = inverse_matrices[instance] * vec4(direction, 0);

#if USE_MBVH
			return instance_meshes[instance]->mbvh->traverse_shadow(new_origin, new_direction, t_min, t_max);
#else
			return instance_meshes[instance]->bvh->traverse_shadow(new_origin, new_direction, t_min, t_max);
#endif
		});
}

int TopLevelBVH::intersect4(float origin_x[4], float origin_y[4], float origin_z[4], float direction_x[4],
							float direction_y[4], float direction_z[4], float t[4], int primID[4], int instID[4],
							float t_min) const
{
	const auto intersection = [&](const int instance, __m128 *inst_mask) {
		const auto &matrix = this->inverse_matrices[instance];

		const simd::vector4 org_x = simd::vector4(origin_x);
		const simd::vector4 org_y = simd::vector4(origin_y);
		const simd::vector4 org_z = simd::vector4(origin_z);
		const simd::vector4 org_w = simd::ONE4;

		const simd::vector4 dir_x = simd::vector4(direction_x);
		const simd::vector4 dir_y = simd::vector4(direction_y);
		const simd::vector4 dir_z = simd::vector4(direction_z);

		const simd::vector4 m0_0 = matrix.matrix[0][0];
		// _mm_shuffle_ps(matrix.cols[0], matrix.cols[0], _MM_SHUFFLE(0, 0, 0, 0));
		const simd::vector4 m0_1 = matrix.matrix[0][1];
		// _mm_shuffle_ps(matrix.cols[0], matrix.cols[0], _MM_SHUFFLE(1, 1, 1, 1));
		const simd::vector4 m0_2 = matrix.matrix[0][2];
		// _mm_shuffle_ps(matrix.cols[0], matrix.cols[0], _MM_SHUFFLE(2, 2, 2, 2));

		simd::vector4 new_origin_x = m0_0 * org_x;
		simd::vector4 new_origin_y = m0_1 * org_x;
		simd::vector4 new_origin_z = m0_2 * org_x;

		simd::vector4 new_direction_x = m0_0 * dir_x;
		simd::vector4 new_direction_y = m0_1 * dir_x;
		simd::vector4 new_direction_z = m0_2 * dir_x;

		const simd::vector4 m1_0 = matrix.matrix[1][0];
		// _mm_shuffle_ps(matrix.cols[1], matrix.cols[1], _MM_SHUFFLE(0, 0, 0, 0));
		const simd::vector4 m1_1 = matrix.matrix[1][1];
		// _mm_shuffle_ps(matrix.cols[1], matrix.cols[1], _MM_SHUFFLE(1, 1, 1, 1));
		const simd::vector4 m1_2 = matrix.matrix[1][2];
		// _mm_shuffle_ps(matrix.cols[1], matrix.cols[1], _MM_SHUFFLE(3, 2, 2, 2));

		new_origin_x += m1_0 * org_y;
		new_origin_y += m1_1 * org_y;
		new_origin_z += m1_2 * org_y;

		new_direction_x += m1_0 * dir_y;
		new_direction_y += m1_1 * dir_y;
		new_direction_z += m1_2 * dir_y;

		const simd::vector4 m2_0 = matrix.matrix[2][0];
		// _mm_shuffle_ps(matrix.cols[2], matrix.cols[2], _MM_SHUFFLE(0, 0, 0, 0));
		const simd::vector4 m2_1 = matrix.matrix[2][1];
		// _mm_shuffle_ps(matrix.cols[2], matrix.cols[2], _MM_SHUFFLE(1, 1, 1, 1));
		const simd::vector4 m2_2 = matrix.matrix[2][2];
		// _mm_shuffle_ps(matrix.cols[2], matrix.cols[2], _MM_SHUFFLE(3, 2, 2, 2));

		new_origin_x += m2_0 * org_z;
		new_origin_y += m2_1 * org_z;
		new_origin_z += m2_2 * org_z;

		new_direction_x += m2_0 * dir_z;
		new_direction_y += m2_1 * dir_z;
		new_direction_z += m2_2 * dir_z;

		const simd::vector4 m3_0 = matrix.matrix[3][0];
		// _mm_shuffle_ps(matrix.cols[3], matrix.cols[3], _MM_SHUFFLE(0, 0, 0, 0));
		const simd::vector4 m3_1 = matrix.matrix[3][1];
		// _mm_shuffle_ps(matrix.cols[3], matrix.cols[3], _MM_SHUFFLE(1, 1, 1, 1));
		const simd::vector4 m3_2 = matrix.matrix[3][2];
		// _mm_shuffle_ps(matrix.cols[3], matrix.cols[3], _MM_SHUFFLE(3, 2, 2, 2));

		new_origin_x += m3_0 * org_w;
		new_origin_y += m3_1 * org_w;
		new_origin_z += m3_2 * org_w;

		const float *ox = reinterpret_cast<float *>(&new_origin_x);
		const float *oy = reinterpret_cast<float *>(&new_origin_y);
		const float *oz = reinterpret_cast<float *>(&new_origin_z);
		const float *dx = reinterpret_cast<float *>(&new_direction_x);
		const float *dy = reinterpret_cast<float *>(&new_direction_y);
		const float *dz = reinterpret_cast<float *>(&new_direction_z);

#if PACKET_MBVH
		return instance_meshes[instance]->mbvh->traverse4(ox, oy, oz, dx, dy, dz, t, primID, t_min, inst_mask);
#else
		return instance_meshes[instance]->bvh->traverse4(ox, oy, oz, dx, dy, dz, t, primID, t_min, inst_mask);
#endif
	};

	__m128 mask = _mm_setzero_ps();
#if TOP_PACKET_MBVH
	return MBVHNode::traverse_mbvh4(origin_x, origin_y, origin_z, direction_x, direction_y, direction_z, t, instID,
									reinterpret_cast<const MBVHNode *>(mbvh.value().nodes), mbvh.value().indices, &mask,
									intersection);
#else
	return BVHNode::traverse_bvh4(origin_x, origin_y, origin_z, direction_x, direction_y, direction_z, t, instID,
								  reinterpret_cast<const BVHNode *>(bvh.value().nodes), bvh.value().indices, &mask,
								  intersection);
#endif
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
		centers.emplace_back();
	}

	aabbs[idx] = boundingBox;
	instance_meshes[idx] = tree;
	matrices[idx] = transform;
	inverse_matrices[idx] = inverse(transform);
	normal_matrices[idx] = mat4(transpose(inverse(mat3(transform))));
	instance_aabbs[idx] = calculate_world_bounds(boundingBox, matrices[idx]);
	centers[idx] = instance_aabbs[idx].center().vec;
}

AABB TopLevelBVH::calculate_world_bounds(const AABB &originalBounds, const simd::matrix4 &matrix)
{
	using namespace simd;

	const vector4 p1 = matrix * vector4(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f);
	const vector4 p5 = matrix * vector4(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f);
	const vector4 p2 = matrix * vector4(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmin[2], 1.f);
	const vector4 p3 = matrix * vector4(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmax[2], 1.f);
	const vector4 p4 = matrix * vector4(originalBounds.bmin[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f);
	const vector4 p6 = matrix * vector4(originalBounds.bmax[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f);
	const vector4 p7 = matrix * vector4(originalBounds.bmin[0], originalBounds.bmax[1], originalBounds.bmin[2], 1.f);
	const vector4 p8 = matrix * vector4(originalBounds.bmax[0], originalBounds.bmin[1], originalBounds.bmax[2], 1.f);

	AABB transformedAABB = {};
	transformedAABB.grow(p1);
	transformedAABB.grow(p2);
	transformedAABB.grow(p3);
	transformedAABB.grow(p4);
	transformedAABB.grow(p5);
	transformedAABB.grow(p6);
	transformedAABB.grow(p7);
	transformedAABB.grow(p8);

	transformedAABB.offset_by(1e-6f);

	return transformedAABB;
}

const rfw::Triangle &TopLevelBVH::get_triangle(int instID, int primID) const
{
	assert(instID > 0 && instID < instance_meshes.size());
	return instance_meshes[instID]->triangles[primID];
}

const rfw::simd::matrix4 &TopLevelBVH::get_normal_matrix(int instID) const { return normal_matrices[instID]; }

const simd::matrix4 &TopLevelBVH::get_inverse_matrix(int instID) const { return inverse_matrices[instID]; }

const rfw::simd::matrix4 &TopLevelBVH::get_instance_matrix(int instID) const { return matrices[instID]; }

} // namespace rfw::bvh