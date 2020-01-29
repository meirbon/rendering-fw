#include "BVH.h"

#include <utils/Timer.h>
#include <utils/Logger.h>

using namespace glm;
using namespace rfw;
using namespace simd;

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
		utils::logger::log("Building MBVH took: %f ms. Poolptr: %i", t.elapsed(), pool_ptr.load());

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

bool MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx,
						glm::vec2 *bary)
{
	return MBVHNode::traverse_mbvh(origin, dir, t_min, ray_t, primIdx, mbvh_nodes.data(), bvh->prim_indices.data(),
								   [&](uint primID) {
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

									   const vec3 q = cross(s, e1);
									   const float v = f * dot(dir, q);
									   if (u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f)
										   return false;

									   const float t = f * dot(e2, q);

									   if (t > t_min && *ray_t > t) // ray intersection
									   {
										   // Barycentrics
										   const vec3 p1 = e1 + p0;
										   const vec3 p2 = e2 + p0;

										   const vec3 p = origin + t * dir;
										   const vec3 N = normalize(cross(e1, e2));
										   const float areaABC = glm::dot(N, cross(e1, e2));
										   const float areaPBC = glm::dot(N, cross(p1 - p, p2 - p));
										   const float areaPCA = glm::dot(N, cross(p2 - p, p0 - p));
										   *bary = glm::vec2(areaPBC / areaABC, areaPCA / areaABC);
										   *ray_t = t;
										   return true;
									   }

									   return false;
								   });
}

bool MBVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx)
{
	const auto intersection = [&](uint primID) {
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

		const vec3 q = cross(s, e1);
		const float v = f * dot(dir, q);
		if (u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f)
			return false;

		const float t = f * dot(e2, q);

		if (t > t_min && *ray_t > t) // ray intersection
		{
			*ray_t = t;
			return true;
		}

		return false;
	};

	return MBVHNode::traverse_mbvh(origin, dir, t_min, ray_t, primIdx, mbvh_nodes.data(), bvh->prim_indices.data(),
								   intersection);
}

int MBVHTree::traverse4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
						const float dir_y[4], const float dir_z[4], float t[4], int primID[4], float t_min,
						__m128 *hit_mask)
{
	const auto t_min4 = vector4(t_min);
	const auto intersection = [&](const int primId, __m128 *store_mask) {
		const vec3 &p0 = bvh->p0s[primId];
		const vec3 &edge1 = bvh->edge1s[primId];
		const vec3 &edge2 = bvh->edge2s[primId];

		const vector4 p0_x = _mm_set1_ps(p0.x);
		const vector4 p0_y = _mm_set1_ps(p0.y);
		const vector4 p0_z = _mm_set1_ps(p0.z);

		const vector4 edge1_x = _mm_set1_ps(edge1.x);
		const vector4 edge1_y = _mm_set1_ps(edge1.y);
		const vector4 edge1_z = _mm_set1_ps(edge1.z);

		const vector4 edge2_x = _mm_set1_ps(edge2.x);
		const vector4 edge2_y = _mm_set1_ps(edge2.y);
		const vector4 edge2_z = _mm_set1_ps(edge2.z);

		// Cross product
		// x = (ay * bz - az * by)
		// y = (az * bx - ax * bz)
		// z = (ax * by - ay * bx)

		vector4 hit_mask = _mm_set_epi32(~0, ~0, ~0, ~0);

		// const vec3 h = cross(dir, edge2);
		const vector4 h_x4 = vector4(dir_y) * edge2_z - vector4(dir_z) * edge2_y;
		const vector4 h_y4 = vector4(dir_z) * edge2_x - vector4(dir_x) * edge2_z;
		const vector4 h_z4 = vector4(dir_x) * edge2_y - vector4(dir_y) * edge2_x;

		// const float a = dot(edge1, h);
		const vector4 a4 = (edge1_x * h_x4) + (edge1_y * h_y4) + (edge1_z * h_z4);
		// if (a > -1e-6f && a < 1e-6f)
		//	return false;
		const vector4 mask_a4 = ((a4 <= vector4(-1e-6f)) | (a4 >= vector4(1e-6f)));
		if (mask_a4.move_mask() == 0)
			return 0;

		hit_mask &= mask_a4;

		// const float f = 1.f / a;
		const vector4 f4 = ONE4 / a4;

		// const vec3 s = org - p0;
		const vector4 s_x4 = vector4(origin_x) - p0_x;
		const vector4 s_y4 = vector4(origin_y) - p0_y;
		const vector4 s_z4 = vector4(origin_z) - p0_z;

		// const float u = f * dot(s, h);
		const vector4 u4 = f4 * ((s_x4 * h_x4) + (s_y4 * h_y4) + (s_z4 * h_z4));

		// if (u < 0.0f || u > 1.0f)
		//	return false;
		const vector4 mask_u = ((u4 >= ZERO4) & (u4 <= ONE4));
		if (mask_u.move_mask() == 0)
			return 0;

		hit_mask &= mask_u;

		// const vec3 q = cross(s, edge1);
		const vector4 q_x4 = (s_y4 * edge1_z) - (s_z4 * edge1_y);
		const vector4 q_y4 = (s_z4 * edge1_x) - (s_x4 * edge1_z);
		const vector4 q_z4 = (s_x4 * edge1_y) - (s_y4 * edge1_x);

		// const float v = f * dot(dir, q);
		const vector4 v4 = f4 * ((vector4(dir_x) * q_x4) + (vector4(dir_y) * q_y4) + (vector4(dir_z) * q_z4));

		// if (v < 0.0f || u + v > 1.0f)
		//	return false;
		const vector4 mask_uv = ((v4 >= ZERO4) & ((u4 + v4) <= ONE4));
		if (mask_uv.move_mask() == 0)
			return 0;

		hit_mask &= mask_uv;

		// const float t = f * dot(edge2, q);
		const vector4 t4 = f4 * ((edge2_x * q_x4) + (edge2_y * q_y4) + (edge2_z * q_z4));

		// if (t > tmin && *rayt > t) // ray intersection
		*store_mask = (((t4 > t_min4) & (vector4(t) > t4)) & hit_mask).vec_4;
		const int storage_mask = _mm_movemask_ps(*store_mask);
		if (storage_mask > 0)
		{
			// *rayt = t;
			t4.write_to(t, *store_mask);
		}

		return storage_mask;
	};

	return MBVHNode::traverse_mbvh4(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, t, primID, mbvh_nodes.data(),
									bvh->prim_indices.data(), hit_mask, intersection);
}

int MBVHTree::traverse4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
						const float dir_y[4], const float dir_z[4], float t[4], float bary_x[4], float bary_y[4],
						int primID[4], float t_min, __m128 *hit_mask)
{

	const auto t_min4 = vector4(t_min);

	const auto intersection = [&](const int primId, __m128 *store_mask) {
		const vec3 &p0 = bvh->p0s[primId];
		const vec3 &edge1 = bvh->edge1s[primId];
		const vec3 &edge2 = bvh->edge2s[primId];

		const vector4 p0_x = _mm_set1_ps(p0.x);
		const vector4 p0_y = _mm_set1_ps(p0.y);
		const vector4 p0_z = _mm_set1_ps(p0.z);

		const vector4 edge1_x = _mm_set1_ps(edge1.x);
		const vector4 edge1_y = _mm_set1_ps(edge1.y);
		const vector4 edge1_z = _mm_set1_ps(edge1.z);

		const vector4 edge2_x = _mm_set1_ps(edge2.x);
		const vector4 edge2_y = _mm_set1_ps(edge2.y);
		const vector4 edge2_z = _mm_set1_ps(edge2.z);

		// Cross product
		// x = (ay * bz - az * by)
		// y = (az * bx - ax * bz)
		// z = (ax * by - ay * bx)

		vector4 hit_mask = _mm_set_epi32(~0, ~0, ~0, ~0);

		// const vec3 h = cross(dir, edge2);
		const vector4 h_x4 = vector4(dir_y) * edge2_z - vector4(dir_z) * edge2_y;
		const vector4 h_y4 = vector4(dir_z) * edge2_x - vector4(dir_x) * edge2_z;
		const vector4 h_z4 = vector4(dir_x) * edge2_y - vector4(dir_y) * edge2_x;

		// const float a = dot(edge1, h);
		const vector4 a4 = (edge1_x * h_x4) + (edge1_y * h_y4) + (edge1_z * h_z4);
		// if (a > -1e-6f && a < 1e-6f)
		//	return false;
		const vector4 mask_a4 = ((a4 <= vector4(-1e-6f)) | (a4 >= vector4(1e-6f)));
		if (mask_a4.move_mask() == 0)
			return 0;

		hit_mask &= mask_a4;

		// const float f = 1.f / a;
		const vector4 f4 = ONE4 / a4;

		// const vec3 s = org - p0;
		const vector4 s_x4 = vector4(origin_x) - p0_x;
		const vector4 s_y4 = vector4(origin_y) - p0_y;
		const vector4 s_z4 = vector4(origin_z) - p0_z;

		// const float u = f * dot(s, h);
		const vector4 u4 = f4 * ((s_x4 * h_x4) + (s_y4 * h_y4) + (s_z4 * h_z4));

		// if (u < 0.0f || u > 1.0f)
		//	return false;
		const vector4 mask_u = ((u4 >= ZERO4) & (u4 <= ONE4));
		if (mask_u.move_mask() == 0)
			return 0;

		hit_mask &= mask_u;

		// const vec3 q = cross(s, edge1);
		const vector4 q_x4 = (s_y4 * edge1_z) - (s_z4 * edge1_y);
		const vector4 q_y4 = (s_z4 * edge1_x) - (s_x4 * edge1_z);
		const vector4 q_z4 = (s_x4 * edge1_y) - (s_y4 * edge1_x);

		// const float v = f * dot(dir, q);
		const vector4 v4 = f4 * ((vector4(dir_x) * q_x4) + (vector4(dir_y) * q_y4) + (vector4(dir_z) * q_z4));

		// if (v < 0.0f || u + v > 1.0f)
		//	return false;
		const vector4 mask_uv = ((v4 >= ZERO4) & ((u4 + v4) <= ONE4));
		if (mask_uv.move_mask() == 0)
			return 0;

		hit_mask &= mask_uv;

		// const float t = f * dot(edge2, q);
		const vector4 t4 = f4 * ((edge2_x * q_x4) + (edge2_y * q_y4) + (edge2_z * q_z4));

		// if (t > tmin && *rayt > t) // ray intersection
		*store_mask = (((t4 > t_min4) & (vector4(t) > t4)) & hit_mask).vec_4;
		const int storage_mask = _mm_movemask_ps(*store_mask);
		if (storage_mask > 0)
		{
			// const vec3 p1 = e1 + p0;
			const vector4 p1_x = edge1_x + p0_x;
			const vector4 p1_y = edge1_y + p0_y;
			const vector4 p1_z = edge1_z + p0_z;

			// const vec3 p2 = e2 + p0;
			const vector4 p2_x = edge2_x + p0_x;
			const vector4 p2_y = edge2_y + p0_y;
			const vector4 p2_z = edge2_z + p0_z;

			// const vec3 p = origin + t * dir;
			const vector4 p_x = vector4(origin_x) + t4 * vector4(dir_x);
			const vector4 p_y = vector4(origin_y) + t4 * vector4(dir_y);
			const vector4 p_z = vector4(origin_z) + t4 * vector4(dir_z);

			const vector4 edge1_cross_edge2_x4 = edge1_y * edge2_z - edge1_z * edge2_y;
			const vector4 edge1_cross_edge2_y4 = edge1_z * edge2_x - edge1_x * edge2_z;
			const vector4 edge1_cross_edge2_z4 = edge1_x * edge2_y - edge1_y * edge2_x;

			// const vec3 N = normalize(cross(e1, e2));
			const vector4 N_length =
				(edge1_cross_edge2_x4 * edge1_cross_edge2_x4 + edge1_cross_edge2_y4 * edge1_cross_edge2_y4 +
				 edge1_cross_edge2_z4 * edge1_cross_edge2_z4)
					.inv_sqrt();
			const vector4 N_x4 = edge1_cross_edge2_x4 * N_length;
			const vector4 N_y4 = edge1_cross_edge2_y4 * N_length;
			const vector4 N_z4 = edge1_cross_edge2_z4 * N_length;

			const vector4 p0_min_p_x = p0_x - p_x;
			const vector4 p0_min_p_y = p0_y - p_y;
			const vector4 p0_min_p_z = p0_z - p_z;

			const vector4 p1_min_p_x = p1_x - p_x;
			const vector4 p1_min_p_y = p1_y - p_y;
			const vector4 p1_min_p_z = p1_z - p_z;

			const vector4 p2_min_p_x = p2_x - p_x;
			const vector4 p2_min_p_y = p2_y - p_y;
			const vector4 p2_min_p_z = p2_z - p_z;

			const vector4 &ABC_x4 = edge1_cross_edge2_x4;
			const vector4 &ABC_y4 = edge1_cross_edge2_y4;
			const vector4 &ABC_z4 = edge1_cross_edge2_z4;

			const vector4 PBC_x4 = p1_min_p_y * p2_min_p_z - p1_min_p_z * p2_min_p_y;
			const vector4 PBC_y4 = p1_min_p_z * p2_min_p_x - p1_min_p_x * p2_min_p_z;
			const vector4 PBC_z4 = p1_min_p_x * p2_min_p_y - p1_min_p_y * p2_min_p_x;

			const vector4 PCA_x4 = p2_min_p_y * p0_min_p_z - p2_min_p_z * p0_min_p_y;
			const vector4 PCA_y4 = p2_min_p_z * p0_min_p_x - p2_min_p_x * p0_min_p_z;
			const vector4 PCA_z4 = p2_min_p_x * p0_min_p_y - p2_min_p_y * p0_min_p_x;

			// const float areaABC = glm::dot(N, cross(e1, e2));
			const vector4 areaABC = N_x4 * ABC_x4 + N_y4 * ABC_y4 + N_z4 * ABC_z4;
			// const float areaPBC = glm::dot(N, cross(p1 - p, p2 - p));
			const vector4 areaPBC = N_x4 * PBC_x4 + N_y4 * PBC_y4 + N_z4 * PBC_z4;
			// const float areaPCA = glm::dot(N, cross(p2 - p, p0 - p));
			const vector4 areaPCA = N_x4 * PCA_x4 + N_y4 * PCA_y4 + N_z4 * PCA_z4;

			// bary = glm::vec2(areaPBC / areaABC, areaPCA / areaABC);
			const vector4 bary_x4 = areaPBC / areaABC;
			const vector4 bary_y4 = areaPCA / areaABC;
			bary_x4.write_to(bary_x, *store_mask);
			bary_y4.write_to(bary_y, *store_mask);

			// *rayt = t;
			t4.write_to(t, *store_mask);
		}

		return storage_mask;
	};

	return MBVHNode::traverse_mbvh4(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, t, primID, mbvh_nodes.data(),
									bvh->prim_indices.data(), hit_mask, intersection);
}

bool MBVHTree::traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max)
{
	const auto intersection = [&](uint primID) {
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

		const vec3 q = cross(s, e1);
		const float v = f * dot(dir, q);

		if (u < 0.0f || u > 1.0f || v < 0.0f || u + v > 1.0f)
			return false;

		const float t = f * dot(e2, q);

		return t > t_min && t_max > t; // ray intersection
	};

	return MBVHNode::traverse_mbvh_shadow(origin, dir, t_min, t_max, mbvh_nodes.data(), bvh->prim_indices.data(),
										  intersection);
}

} // namespace rfw::bvh