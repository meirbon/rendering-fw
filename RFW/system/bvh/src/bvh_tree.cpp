#include <bvh/bvh.h>

#include <rfw/utils/timer.h>
#include <rfw/utils/logger.h>

#include <atomic>
#include <tbb/parallel_for.h>

#include <rtbvh.hpp>

using namespace glm;
using namespace rfw;

using namespace simd;

namespace rfw::bvh
{

BVHTree::BVHTree()
{
	static_assert(sizeof(rtbvh::RTAabb) == sizeof(AABB));
	static_assert(sizeof(rtbvh::RTBvhNode) == sizeof(BVHNode));
};

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount) : vertex_count(vertexCount), face_count(vertexCount / 3)
{
	indices = nullptr;
	set_vertices(vertices);
}

BVHTree::BVHTree(const glm::vec4 *vertices, int vertexCount, const glm::uvec3 *indices, int faceCount)
	: vertex_count(vertexCount), face_count(faceCount)
{
	set_vertices(vertices, indices);
}

BVHTree::~BVHTree() { reset(); }

void BVHTree::reset()
{
	if (instance.has_value())
	{
		rtbvh::free_bvh(instance.value());
		instance = std::nullopt;
	}
}

void BVHTree::construct(Type type)
{
	reset();
	std::vector<glm::vec4> centers(face_count);
	tbb::parallel_for(0, face_count,
					  [&](int i)
					  {
						  if (indices)
						  {
							  const auto i0 = indices[i].x;
							  const auto i1 = indices[i].y;
							  const auto i2 = indices[i].z;
							  centers[i] = (vertices[i0] + vertices[i1] + vertices[i2]) * (1.0f / 3.0f);
						  }
						  else
						  {
							  const auto i0 = i * 3 + 0;
							  const auto i1 = i * 3 + 1;
							  const auto i2 = i * 3 + 2;
							  centers[i] = (vertices[i0] + vertices[i1] + vertices[i2]) * (1.0f / 3.0f);
						  }
					  });

	rtbvh::RTBvh bvh{};
	rtbvh::ResultCode error = rtbvh::ResultCode::Ok;
	switch (type)
	{
	case Type::BinnedSAH:
		error = rtbvh::create_bvh(reinterpret_cast<const rtbvh::RTAabb *>(aabbs.data()), face_count,
								  reinterpret_cast<const float *>(centers.data()), sizeof(glm::vec4), 1,
								  rtbvh::BvhType::BinnedSAH, &bvh);
		instance = std::make_optional(bvh);
		break;
	case Type::LocallyOrderedClustering:
		error = rtbvh::create_bvh(reinterpret_cast<const rtbvh::RTAabb *>(aabbs.data()), face_count,
								  reinterpret_cast<const float *>(centers.data()), sizeof(glm::vec4), 1,
								  rtbvh::BvhType::LocallyOrderedClustered, &bvh);
		instance = std::make_optional(bvh);
		break;
	case Type::SpatialSAH:
	{
		const glm::vec4 *verts = vertices;
		if (indices)
		{
			// Use inlined vertices instead of indexed
			verts = splat_vertices.data();
		}

		error = rtbvh::create_spatial_Bvh(reinterpret_cast<const rtbvh::RTAabb *>(aabbs.data()), face_count,
										  reinterpret_cast<const float *>(centers.data()), sizeof(glm::vec4),
										  reinterpret_cast<const float *>(verts), sizeof(glm::vec4),
										  3 * sizeof(glm::vec4), 1, &bvh);
		instance = std::make_optional(bvh);
		break;
	}

	default:
		break;
	}

	assert(error == rtbvh::ResultCode::Ok);
}

void BVHTree::refit(const glm::vec4 *vertices)
{
	set_vertices(vertices);
	rtbvh::refit(reinterpret_cast<const rtbvh::RTAabb *>(aabbs.data()), instance.value());
}

void BVHTree::refit(const glm::vec4 *vertices, const glm::uvec3 *indices)
{
	set_vertices(vertices, indices);
	rtbvh::refit(reinterpret_cast<const rtbvh::RTAabb *>(aabbs.data()), instance.value());
}

bool BVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx,
					   glm::vec2 *bary)
{
	return BVHNode::traverse_bvh(origin, dir, t_min, ray_t, primIdx,
								 reinterpret_cast<const BVHNode *>(instance.value().nodes), instance.value().indices,
								 [&](uint primID)
								 {
									 const vec3 &p0 = p0s[primID];
									 const vec3 &e1 = edge1s[primID];
									 const vec3 &e2 = edge2s[primID];
									 const vec3 h = cross(dir, e2);

									 const float a = dot(e1, h);
									 if (a > -1e-6f && a < 1e-6f)
										 return false;

									 const float f = 1.f / a;
									 const vec3 s = origin - p0;
									 const float u = f * dot(s, h);
									 if (u < 0.0f || u > 1.0f)
										 return false;

									 const vec3 q = cross(s, e1);
									 const float v = f * dot(dir, q);
									 if (v < 0.0f || u + v > 1.0f)
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

bool BVHTree::traverse(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float *ray_t, int *primIdx)
{
	const auto intersection = [&](uint primID)
	{
		const vec3 &p0 = p0s[primID];
		const vec3 &e1 = edge1s[primID];
		const vec3 &e2 = edge2s[primID];
		const vec3 h = cross(dir, e2);

		const float a = dot(e1, h);
		if (a > -1e-6f && a < 1e-6f)
			return false;

		const float f = 1.f / a;
		const vec3 s = origin - p0;
		const float u = f * dot(s, h);
		if (u < 0.0f || u > 1.0f)
			return false;

		const vec3 q = cross(s, e1);
		const float v = f * dot(dir, q);
		if (v < 0.0f || u + v > 1.0f)
			return false;

		const float t = f * dot(e2, q);

		if (t > t_min && *ray_t > t) // ray intersection
		{
			*ray_t = t;
			return true;
		}

		return false;
	};

	return BVHNode::traverse_bvh(origin, dir, t_min, ray_t, primIdx,
								 reinterpret_cast<const BVHNode *>(instance.value().nodes), instance.value().indices,
								 intersection);
}
int BVHTree::traverse4(const float origin_x[4], const float origin_y[4], const float origin_z[4], const float dir_x[4],
					   const float dir_y[4], const float dir_z[4], float t[4], int primID[4], float t_min,
					   __m128 *hit_mask)
{
	const auto intersection = [&](const int primId, __m128 *store_mask)
	{
		const vec3 &p0 = p0s[primId];
		const vec3 &edge1 = edge1s[primId];
		const vec3 &edge2 = edge2s[primId];

#define PER_RAY 0
#if PER_RAY
		bool result[4] = {false, false, false, false};

		const auto t_intersect = [&](uint primID, vec3 org, vec3 dir, float *ray_t)
		{
			const vec3 h = cross(dir, edge2);

			const float a = dot(edge1, h);
			if (a > -1e-6f && a < 1e-6f)
				return false;

			const float f = 1.f / a;
			const vec3 s = org - p0;
			const float u = f * dot(s, h);
			if (u < 0.0f || u > 1.0f)
				return false;

			const vec3 q = cross(s, edge1);
			const float v = f * dot(dir, q);
			if (v < 0.0f || u + v > 1.0f)
				return false;

			const float t = f * dot(edge2, q);

			if (t > t_min && *ray_t > t) // ray intersection
			{
				*ray_t = t;
				return true;
			}

			return false;
		};

		result[0] =
			t_intersect(primId, vec3(origin_x[0], origin_y[0], origin_z[0]), vec3(dir_x[0], dir_y[0], dir_z[0]), &t[0]);
		result[1] =
			t_intersect(primId, vec3(origin_x[1], origin_y[1], origin_z[1]), vec3(dir_x[1], dir_y[1], dir_z[1]), &t[1]);
		result[2] =
			t_intersect(primId, vec3(origin_x[2], origin_y[2], origin_z[2]), vec3(dir_x[2], dir_y[2], dir_z[2]), &t[2]);
		result[3] =
			t_intersect(primId, vec3(origin_x[3], origin_y[3], origin_z[3]), vec3(dir_x[3], dir_y[3], dir_z[3]), &t[3]);

		*store_mask = _mm_castsi128_ps(
			_mm_set_epi32(result[3] ? ~0 : 0, result[2] ? ~0 : 0, result[1] ? ~0 : 0, result[0] ? ~0 : 0));

		int res = 0;
		res |= result[0] ? 1 : 0;
		res |= result[1] ? 2 : 0;
		res |= result[2] ? 4 : 0;
		res |= result[3] ? 8 : 0;
		return res;
#else
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
		*store_mask = (((t4 > ZERO4) & (vector4(t) > t4)) & hit_mask).vec_4;
		const int storage_mask = _mm_movemask_ps(*store_mask);
		if (storage_mask > 0)
		{
			// *rayt = t;
			t4.write_to(t, *store_mask);
		}

		return storage_mask;
#endif
	};

	return BVHNode::traverse_bvh4(origin_x, origin_y, origin_z, dir_x, dir_y, dir_z, t, primID,
								  reinterpret_cast<const BVHNode *>(instance.value().nodes), instance.value().indices,
								  hit_mask, intersection);
}

bool BVHTree::traverse_shadow(const glm::vec3 &origin, const glm::vec3 &dir, float t_min, float t_max)
{
	return BVHNode::traverse_bvh_shadow(
		origin, dir, t_min, t_max, reinterpret_cast<const BVHNode *>(instance.value().nodes), instance.value().indices,
		[&](uint primID)
		{
			const vec3 &p0 = p0s[primID];
			const vec3 &e1 = edge1s[primID];
			const vec3 &e2 = edge2s[primID];

			const vec3 h = cross(dir, e2);

			const float a = dot(e1, h);
			if (a > -1e-6f && a < 1e-6f)
				return false;

			const float f = 1.f / a;
			const vec3 s = origin - p0;
			const float u = f * dot(s, h);
			if (u < 0.0f || u > 1.0f)
				return false;

			const vec3 q = cross(s, e1);
			const float v = f * dot(dir, q);
			if (v < 0.0f || u + v > 1.0f)
				return false;

			const float t = f * dot(e2, q);

			if (t > t_min && t_max > t) // ray intersection
				return true;

			return false;
		});
}

void BVHTree::set_vertices(const glm::vec4 *verts)
{
	vertices = verts;
	indices = nullptr;

	// Recalculate data
	aabbs.resize(face_count);
	p0s.resize(face_count);
	edge1s.resize(face_count);
	edge2s.resize(face_count);

	for (int i = 0; i < face_count; i++)
	{
		const uvec3 idx = uvec3(i * 3) + uvec3(0, 1, 2);

		const vec3 p0 = vec3(vertices[idx.x]);
		const vec3 p1 = vec3(vertices[idx.y]);
		const vec3 p2 = vec3(vertices[idx.z]);

		aabbs[i] = AABB::invalid();
		aabbs[i].grow(p0);
		aabbs[i].grow(p1);
		aabbs[i].grow(p2);
		aabbs[i].offset_by(1e-5f);

		p0s[i] = p0;
		edge1s[i] = p1 - p0;
		edge2s[i] = p2 - p0;
	}
}

void BVHTree::set_vertices(const glm::vec4 *verts, const glm::uvec3 *ids)
{
	vertices = verts;
	indices = ids;

	// Recalculate data
	aabbs.resize(face_count);
	p0s.resize(face_count);
	edge1s.resize(face_count);
	edge2s.resize(face_count);
	splat_vertices.resize(face_count * 3);

	tbb::parallel_for(0, face_count,
					  [&](int i)
					  {
						  const uvec3 &idx = indices[i];

						  const vec3 p0 = vec3(vertices[idx.x]);
						  const vec3 p1 = vec3(vertices[idx.y]);
						  const vec3 p2 = vec3(vertices[idx.z]);

						  splat_vertices[i * 3 + 0] = vec4(p0, 1.0f);
						  splat_vertices[i * 3 + 1] = vec4(p1, 1.0f);
						  splat_vertices[i * 3 + 2] = vec4(p2, 1.0f);

						  aabbs[i] = AABB::invalid();
						  aabbs[i].grow(p0);
						  aabbs[i].grow(p1);
						  aabbs[i].grow(p2);
						  aabbs[i].offset_by(1e-5f);

						  p0s[i] = p0;
						  edge1s[i] = p1 - p0;
						  edge2s[i] = p2 - p0;
					  });
}

AABB BVHTree::get_aabb() const { return reinterpret_cast<const BVHNode *>(instance.value().nodes)[0].bounds; }

} // namespace rfw::bvh