#include "PCH.h"
#include "Traversal.h"

using namespace rfw;
using namespace simd;

inline int intersect4(cpurt::RayPacket4 &packet, const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2, __m128 *store_mask, float epsilon)
{
	const vector4 p0_x = _mm_set1_ps(p0.x);
	const vector4 p0_y = _mm_set1_ps(p0.y);
	const vector4 p0_z = _mm_set1_ps(p0.z);

	const vector4 edge1 = vector4(p1) - p0;
	const vector4 edge2 = vector4(p2) - p0;

	const vector4 edge1_x = _mm_set1_ps(edge1[0]);
	const vector4 edge1_y = _mm_set1_ps(edge1[1]);
	const vector4 edge1_z = _mm_set1_ps(edge1[2]);

	const vector4 edge2_x = _mm_set1_ps(edge2[0]);
	const vector4 edge2_y = _mm_set1_ps(edge2[1]);
	const vector4 edge2_z = _mm_set1_ps(edge2[2]);

	// Cross product
	// x = (ay * bz - az * by)
	// y = (az * bx - ax * bz)
	// z = (ax * by - ay * bx)
	// const vec3 h = cross(dir, edge2);

	const vector4 h_x4 = packet.direction_y4[0] * edge2_z - packet.direction_z4[0] * edge2_y;
	const vector4 h_y4 = packet.direction_z4[0] * edge2_x - packet.direction_x4[0] * edge2_z;
	const vector4 h_z4 = packet.direction_x4[0] * edge2_y - packet.direction_y4[0] * edge2_x;

	// const float a = dot(edge1, h);
	const vector4 a4 = (edge1_x * h_x4) + (edge1_y * h_y4) + (edge1_z * h_z4);

	// if (a > -epsilon && a < epsilon)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_a4 = (a4 <= vector4(-epsilon)) | (a4 >= vector4(epsilon));
	*store_mask = mask_a4.vec_4;
	int mask = mask_a4.move_mask();
	if (mask == 0)
		return 0;

	// const float f = 1.f / a;
	const vector4 f4 = ONE4 / a4;

	// const vec3 s = org - p0;
	const vector4 s_x4 = packet.origin_x4[0] - p0_x;
	const vector4 s_y4 = packet.origin_y4[0] - p0_y;
	const vector4 s_z4 = packet.origin_z4[0] - p0_z;

	// const float u = f * dot(s, h);
	const vector4 u4 = f4 * ((s_x4 * h_x4) + (s_y4 * h_y4) + (s_z4 * h_z4));

	// if (u < 0.0f || u > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_u4 = (u4 >= ZERO4) & (u4 <= ONE4);
	*store_mask = ((*store_mask) & mask_u4).vec_4;
	mask &= mask_u4.move_mask();
	if (mask == 0)
		return 0;

	// const vec3 q = cross(s, edge1);
	const vector4 q_x4 = (s_y4 * edge1_z) - (s_z4 * edge1_y);
	const vector4 q_y4 = (s_z4 * edge1_x) - (s_x4 * edge1_z);
	const vector4 q_z4 = (s_x4 * edge1_y) - (s_y4 * edge1_x);

	// const float v = f * dot(dir, q);
	const vector4 v4 = f4 * ((packet.direction_x4[0] * q_x4) + (packet.direction_y4[0] * q_y4) + (packet.direction_z4[0] * q_z4));

	// if (v < 0.0f || u + v > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_vu4 = (v4 >= ZERO4) & ((u4 + v4) <= ONE4);

	*store_mask = (*store_mask & mask_vu4).vec_4;
	mask &= mask_vu4.move_mask();
	if (mask == 0)
		return 0;

	// const float t = f * dot(edge2, q);
	const vector4 t4 = f4 * ((edge2_x * q_x4) + (edge2_y * q_y4) + (edge2_z * q_z4));

	// if (t > tmin && *rayt > t) // ray intersection
	*store_mask = (*store_mask & ((t4 > epsilon) & (packet.t4[0] > t4))).vec_4;
	const int storage_mask = _mm_movemask_ps(*store_mask);
	if (storage_mask > 0)
	{
		//	*rayt = t;
		t4.write_to(packet.t, vector4(*store_mask));
		return storage_mask;
	}

	return 0;
}

inline int intersect4(cpurt::RayPacket4 &packet, const glm::vec3 &p0, const glm::vec3 &edge1, const glm::vec3 &edge2, __m128 *store_mask, float epsilon)
{
	const vector4 p0_x = _mm_set1_ps(p0.x);
	const vector4 p0_y = _mm_set1_ps(p0.y);
	const vector4 p0_z = _mm_set1_ps(p0.z);

	const vector4 edge1_x = _mm_set1_ps(edge1[0]);
	const vector4 edge1_y = _mm_set1_ps(edge1[1]);
	const vector4 edge1_z = _mm_set1_ps(edge1[2]);

	const vector4 edge2_x = _mm_set1_ps(edge2[0]);
	const vector4 edge2_y = _mm_set1_ps(edge2[1]);
	const vector4 edge2_z = _mm_set1_ps(edge2[2]);

	// Cross product
	// x = (ay * bz - az * by)
	// y = (az * bx - ax * bz)
	// z = (ax * by - ay * bx)
	// const vec3 h = cross(dir, edge2);

	const vector4 h_x4 = packet.direction_y4[0] * edge2_z - packet.direction_z4[0] * edge2_y;
	const vector4 h_y4 = packet.direction_z4[0] * edge2_x - packet.direction_x4[0] * edge2_z;
	const vector4 h_z4 = packet.direction_x4[0] * edge2_y - packet.direction_y4[0] * edge2_x;

	// const float a = dot(edge1, h);
	const vector4 a4 = (edge1_x * h_x4) + (edge1_y * h_y4) + (edge1_z * h_z4);

	// if (a > -epsilon && a < epsilon)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_a4 = (a4 <= vector4(-epsilon)) | (a4 >= vector4(epsilon));
	*store_mask = mask_a4.vec_4;
	int mask = mask_a4.move_mask();
	if (mask == 0)
		return 0;

	// const float f = 1.f / a;
	const vector4 f4 = ONE4 / a4;

	// const vec3 s = org - p0;
	const vector4 s_x4 = packet.origin_x4[0] - p0_x;
	const vector4 s_y4 = packet.origin_y4[0] - p0_y;
	const vector4 s_z4 = packet.origin_z4[0] - p0_z;

	// const float u = f * dot(s, h);
	const vector4 u4 = f4 * ((s_x4 * h_x4) + (s_y4 * h_y4) + (s_z4 * h_z4));

	// if (u < 0.0f || u > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_u4 = (u4 >= ZERO4) & (u4 <= ONE4);
	*store_mask = ((*store_mask) & mask_u4).vec_4;
	mask &= mask_u4.move_mask();
	if (mask == 0)
		return 0;

	// const vec3 q = cross(s, edge1);
	const vector4 q_x4 = (s_y4 * edge1_z) - (s_z4 * edge1_y);
	const vector4 q_y4 = (s_z4 * edge1_x) - (s_x4 * edge1_z);
	const vector4 q_z4 = (s_x4 * edge1_y) - (s_y4 * edge1_x);

	// const float v = f * dot(dir, q);
	const vector4 v4 = f4 * ((packet.direction_x4[0] * q_x4) + (packet.direction_y4[0] * q_y4) + (packet.direction_z4[0] * q_z4));

	// if (v < 0.0f || u + v > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const vector4 mask_vu4 = (v4 >= ZERO4) & ((u4 + v4) <= ONE4);

	*store_mask = (*store_mask & mask_vu4).vec_4;
	mask &= mask_vu4.move_mask();
	if (mask == 0)
		return 0;

	// const float t = f * dot(edge2, q);
	const vector4 t4 = f4 * ((edge2_x * q_x4) + (edge2_y * q_y4) + (edge2_z * q_z4));

	// if (t > tmin && *rayt > t) // ray intersection
	*store_mask = (*store_mask & ((t4 > epsilon) & (packet.t4[0] > t4))).vec_4;
	const int storage_mask = _mm_movemask_ps(*store_mask);
	if (storage_mask > 0)
	{
		//	*rayt = t;
		t4.write_to(packet.t, vector4(*store_mask));
		return storage_mask;
	}

	return 0;
}
//
// int traverse_mbvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::MBVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
//				  const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask)
//{
//	rfw::bvh::MBVHTraversal todo[32];
//	int stackptr = 0;
//	int hitMask = 0;
//
//	todo[0].leftFirst = 0;
//	todo[0].count = -1;
//
//	*hit_mask = _mm_setzero_ps();
//
//	while (stackptr >= 0)
//	{
//		const int leftFirst = todo[stackptr].leftFirst;
//		const int count = todo[stackptr].count;
//		stackptr--;
//
//		if (count > -1) // leaf node
//		{
//			for (int i = 0; i < count; i++)
//			{
//#if 0 // Triangle intersection debugging
//				int mask[4] = {0, 0, 0, 0};
//				const auto primIdx = primIndices[leftFirst + i];
//				for (int k = 0; k < 4; k++)
//				{
//					const vec3 org = vec3(packet.origin_x[k], packet.origin_y[k], packet.origin_z[k]);
//					const vec3 dir = vec3(packet.direction_x[k], packet.direction_y[k], packet.direction_z[k]);
//
//					if (triangle::intersect_opt(org, dir, t_min, &packet.t[k], p0s[primIdx], edge1s[primIdx], edge2s[primIdx]))
//					{
//						mask[k] = ~0;
//						packet.primID[k] = primIdx;
//					}
//				}
//
//				*hit_mask = _mm_or_ps(*hit_mask, _mm_castsi128_ps(_mm_setr_epi32(mask[0], mask[1], mask[2], mask[3])));
//#else
//				const auto primIdx = primIndices[leftFirst + i];
//				__m128 store_mask = _mm_setzero_ps();
//				hitMask |= intersect4(packet, p0s[primIdx], edge1s[primIdx], edge2s[primIdx], &store_mask);
//				*hit_mask = _mm_or_ps(*hit_mask, store_mask);
//				_mm_maskstore_epi32(packet.primID, _mm_castps_si128(store_mask), _mm_set1_epi32(primIdx));
//#endif
//			}
//			continue;
//		}
//
//
//
//		const rfw::bvh::MBVHHit hit = nodes[leftFirst].intersect4(packet, t_min);
//		for (int i = 3; i >= 0; i--)
//		{ // reversed order, we want to check best nodes first
//			const int idx = (hit.tmini[i] & 0b11);
//			if (hit.result[idx] == 1)
//			{
//				stackptr++;
//				todo[stackptr].leftFirst = nodes[leftFirst].childs[idx];
//				todo[stackptr].count = nodes[leftFirst].counts[idx];
//			}
//		}
//	}
//
//	return hitMask;
//}

inline int intersect4(cpurt::RayPacket4 &packet, const glm::vec3 &p0, const glm::vec3 &edge1, const glm::vec3 &edge2, glm::vec2 *bary4, __m128 *store_mask,
					  float epsilon)
{
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128 zero4 = _mm_setzero_ps();

	const __m128 p0_x = _mm_set1_ps(p0.x);
	const __m128 p0_y = _mm_set1_ps(p0.y);
	const __m128 p0_z = _mm_set1_ps(p0.z);

	const __m128 edge1_x = _mm_set1_ps(edge1.x);
	const __m128 edge1_y = _mm_set1_ps(edge1.y);
	const __m128 edge1_z = _mm_set1_ps(edge1.z);

	const __m128 edge2_x = _mm_set1_ps(edge2.x);
	const __m128 edge2_y = _mm_set1_ps(edge2.y);
	const __m128 edge2_z = _mm_set1_ps(edge2.z);

	// Cross product
	// x = (ay * bz - az * by)
	// y = (az * bx - ax * bz)
	// z = (ax * by - ay * bx)
	// const vec3 h = cross(dir, edge2);
	const __m128 h_x4 = _mm_sub_ps(_mm_mul_ps(packet.direction_y4[0], edge2_z), _mm_mul_ps(packet.direction_z4[0], edge2_y));
	const __m128 h_y4 = _mm_sub_ps(_mm_mul_ps(packet.direction_z4[0], edge2_x), _mm_mul_ps(packet.direction_x4[0], edge2_z));
	const __m128 h_z4 = _mm_sub_ps(_mm_mul_ps(packet.direction_x4[0], edge2_y), _mm_mul_ps(packet.direction_y4[0], edge2_x));

	// const float a = dot(edge1, h);
	const __m128 a4 = _mm_add_ps(_mm_mul_ps(edge1_x, h_x4), _mm_add_ps(_mm_mul_ps(edge1_y, h_y4), _mm_mul_ps(edge1_z, h_z4)));

	// if (a > -epsilon && a < epsilon)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_a4 = _mm_or_ps(_mm_cmple_ps(a4, _mm_set1_ps(-epsilon)), _mm_cmpge_ps(a4, _mm_set1_ps(epsilon)));
	*store_mask = mask_a4;
	int mask = _mm_movemask_ps(mask_a4);
	if (mask == 0)
		return 0;

	// const float f = 1.f / a;
	const __m128 f4 = _mm_div_ps(one4, a4);

	// const vec3 s = org - p0;
	const __m128 s_x4 = _mm_sub_ps(packet.origin_x4[0], p0_x);
	const __m128 s_y4 = _mm_sub_ps(packet.origin_y4[0], p0_y);
	const __m128 s_z4 = _mm_sub_ps(packet.origin_z4[0], p0_z);

	// const float u = f * dot(s, h);
	const __m128 u4 = _mm_mul_ps(f4, _mm_add_ps(_mm_mul_ps(s_x4, h_x4), _mm_add_ps(_mm_mul_ps(s_y4, h_y4), _mm_mul_ps(s_z4, h_z4))));

	// if (u < 0.0f || u > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_u4 = _mm_and_ps(_mm_cmpge_ps(u4, zero4), _mm_cmple_ps(u4, one4));
	*store_mask = _mm_and_ps(*store_mask, mask_u4);
	mask = _mm_movemask_ps(mask_u4);
	if (mask == 0)
		return 0;

	// const vec3 q = cross(s, edge1);
	const __m128 q_x4 = _mm_sub_ps(_mm_mul_ps(s_y4, edge1_z), _mm_mul_ps(s_z4, edge1_y));
	const __m128 q_y4 = _mm_sub_ps(_mm_mul_ps(s_z4, edge1_x), _mm_mul_ps(s_x4, edge1_z));
	const __m128 q_z4 = _mm_sub_ps(_mm_mul_ps(s_x4, edge1_y), _mm_mul_ps(s_y4, edge1_x));

	// const float v = f * dot(dir, q);
	const __m128 dir_dot_q4 =
		_mm_add_ps(_mm_mul_ps(packet.direction_x4[0], q_x4), _mm_add_ps(_mm_mul_ps(packet.direction_y4[0], q_y4), _mm_mul_ps(packet.direction_z4[0], q_z4)));
	const __m128 v4 = _mm_mul_ps(f4, dir_dot_q4);

	// if (v < 0.0f || u + v > 1.0f)
	//	return false;
	// Inverse check, we need to check whether any ray might hit this triangle
	const __m128 mask_vu4 = _mm_and_ps(_mm_cmpge_ps(v4, zero4), _mm_cmple_ps(_mm_add_ps(u4, v4), one4));
	*store_mask = _mm_and_ps(*store_mask, mask_vu4);
	mask = _mm_movemask_ps(mask_vu4);
	if (mask == 0)
		return 0;

	// const float t = f * dot(edge2, q);
	const __m128 t4 = _mm_mul_ps(f4, _mm_add_ps(_mm_mul_ps(edge2_x, q_x4), _mm_add_ps(_mm_mul_ps(edge2_y, q_y4), _mm_mul_ps(edge2_z, q_z4))));

	// if (t > tmin && *rayt > t) // ray intersection
	*store_mask = _mm_and_ps(*store_mask, _mm_and_ps(_mm_cmpgt_ps(t4, _mm_set1_ps(epsilon)), _mm_cmpgt_ps(packet.t4[0], t4)));
	const int storage_mask = _mm_movemask_ps(*store_mask);
	if (storage_mask > 0)
	{
		const vec3 p1 = edge1 + p0;
		const vec3 p2 = edge2 + p0;

		// TODO
		// const rfw::simd::vector4 p_x = packet.origin_x4[0] * packet.t4[0];
		//
		// const vec3 p = origin + t * dir;
		// const vec3 N = normalize(cross(e1, e2));
		// const float areaABC = glm::dot(N, cross(edge1, edge2));
		// const float areaPBC = glm::dot(N, cross(p1 - p, p2 - p));
		// const float areaPCA = glm::dot(N, cross(p2 - p, p0 - p));
		//*bary = glm::vec2(areaPBC / areaABC, areaPCA / areaABC);

		_mm_maskstore_ps(packet.t, _mm_castps_si128(*store_mask), t4);
		return storage_mask;
	}

	return 0;
}

inline int intersect4(cpurt::RayPacket4 &packet, const rfw::bvh::BVHNode &node)
{
	static const __m128 one4 = _mm_set1_ps(1.0f);

	// const __m128 origin = _mm_maskload_ps(value_ptr(org), _mm_set_epi32(0, ~0, ~0, ~0));
	// const __m128 dirInv = _mm_maskload_ps(value_ptr(dirInverse), _mm_set_epi32(0, ~0, ~0, ~0));
	const __m128 inv_direction_x4 = _mm_div_ps(one4, packet.direction_x4[0]);
	const __m128 inv_direction_y4 = _mm_div_ps(one4, packet.direction_y4[0]);
	const __m128 inv_direction_z4 = _mm_div_ps(one4, packet.direction_z4[0]);

	// const glm::vec3 t1 = (glm::make_vec3(bounds.bmin) - org) * dirInverse;
	const __m128 t1_4_x = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmin[0]), packet.origin_x4[0]), inv_direction_x4);
	const __m128 t1_4_y = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmin[1]), packet.origin_y4[0]), inv_direction_y4);
	const __m128 t1_4_z = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmin[2]), packet.origin_z4[0]), inv_direction_z4);

	// const glm::vec3 t2 = (glm::make_vec3(bounds.bmax) - org) * dirInverse;
	const __m128 t2_4_x = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmax[0]), packet.origin_x4[0]), inv_direction_x4);
	const __m128 t2_4_y = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmax[1]), packet.origin_y4[0]), inv_direction_y4);
	const __m128 t2_4_z = _mm_mul_ps(_mm_sub_ps(_mm_set1_ps(node.bounds.bmax[2]), packet.origin_z4[0]), inv_direction_z4);

	// const glm::vec3 min = glm::min(t1, t2);
	const __m128 tmin_x4 = _mm_min_ps(t1_4_x, t2_4_x);
	const __m128 tmin_y4 = _mm_min_ps(t1_4_y, t2_4_y);
	const __m128 tmin_z4 = _mm_min_ps(t1_4_z, t2_4_z);

	// const glm::vec3 max = glm::max(t1, t2);
	const __m128 tmax_x4 = _mm_max_ps(t1_4_x, t2_4_x);
	const __m128 tmax_y4 = _mm_max_ps(t1_4_y, t2_4_y);
	const __m128 tmax_z4 = _mm_max_ps(t1_4_z, t2_4_z);

	//*t_min = glm::max(min.x, glm::max(min.y, min.z));
	__m128 tmin_4 = _mm_max_ps(tmin_x4, _mm_max_ps(tmin_y4, tmin_z4));
	//*t_max = glm::min(max.x, glm::min(max.y, max.z));
	__m128 tmax_4 = _mm_min_ps(tmax_x4, _mm_min_ps(tmax_y4, tmax_z4));

	// return (*tmax) > (*tmin) && (*tmin) < t;
	const __m128 mask_4 = _mm_and_ps(_mm_cmpge_ps(tmax_4, tmin_4), _mm_cmplt_ps(tmin_4, packet.t4[0]));

	return _mm_movemask_ps(mask_4) > 0;
}

// static int intersect4(cpurt::RayPacket4 &packet, const rfw::bvh::MBVHNode &node, __m128 *hit_mask) {}

inline int traverse_bvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices,
						const rfw::bvh::rfwMesh **meshes, __m128 *hit_mask)
{
	bool valid = false;
	rfw::bvh::BVHTraversal todo[32];
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
				const auto &mesh = *meshes[primIdx];

				const int mask = traverse_bvh(packet, t_min, mesh.bvh->bvh_nodes.data(), mesh.bvh->prim_indices.data(), mesh.bvh->p0s.data(),
											  mesh.bvh->edge1s.data(), mesh.bvh->edge2s.data(), &store_mask);
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
			const int hit_left = intersect4(packet, nodes[node.get_left_first()]);
			const int hit_right = intersect4(packet, nodes[node.get_left_first() + 1]);

			if (hit_left && hit_right)
			{
				// TODO: Convert this to use t-value
				if (hit_left > hit_right)
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
			else if (hit_left)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hit_right)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return hitMask;
}

inline int traverse_bvh(cpurt::RayPacket4 &packet, float t_min, const rfw::bvh::BVHNode *nodes, const unsigned int *primIndices, const glm::vec3 *p0s,
						const glm::vec3 *edge1s, const glm::vec3 *edge2s, __m128 *hit_mask)
{
	bool valid = false;
	rfw::bvh::BVHTraversal todo[32];
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
				const int mask = intersect4(packet, p0s[primIdx], edge1s[primIdx], edge2s[primIdx], &store_mask);
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
			const int hit_left = intersect4(packet, nodes[node.get_left_first()]);
			const int hit_right = intersect4(packet, nodes[node.get_left_first() + 1]);

			if (hit_left && hit_right)
			{
				// TODO: Convert this to use t-value
				if (hit_left > hit_right)
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
			else if (hit_left)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first()};
			}
			else if (hit_right)
			{
				stackPtr++;
				todo[stackPtr] = {node.get_left_first() + 1};
			}
		}
	}

	return hitMask;
}
