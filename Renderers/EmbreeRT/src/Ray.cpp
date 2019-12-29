#include "Ray.h"

Ray::CameraParams::CameraParams(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height)
{
	pos_lensSize = glm::vec4(view.pos, view.aperture);
	right_spreadAngle = vec4(view.p2 - view.p1, view.spreadAngle);
	up = vec4(view.p3 - view.p1, 1.0f);
	p1 = vec4(view.p1, 1.0f);

	samplesTaken = samples;
	geometryEpsilon = 1e-5f;
	scrwidth = width;
	scrheight = height;
}

Ray Ray::generateFromView(const Ray::CameraParams &camera, int x, int y, float r0, float r1, float r2, float r3)
{
	Ray ray;
	const float blade = int(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	float bladeParam = blade * piOver4point5;
	x1 = cos(bladeParam);
	y1 = sin(bladeParam);
	bladeParam = (blade + 1.0f) * piOver4point5;
	x2 = cos(bladeParam);
	y2 = sin(bladeParam);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;
	const vec3 p1 = camera.p1;
	const vec3 right = camera.right_spreadAngle;
	const vec3 up = camera.up;

	ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	const vec3 pointOnPixel = p1 + u * right + v * up;
	ray.direction = normalize(pointOnPixel - ray.origin);

	return ray;
}

RTCRayHit4 Ray::GenerateRay4(const CameraParams &camera, const int x[4], const int y[4], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit4 query;

	for (int i = 0; i < 4; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	const __m128 one4 = _mm_set1_ps(1.0f);

	union {
		__m128 r04;
		float r0[4];
	};
	union {
		__m128 r14;
		float r1[4];
	};
	union {
		__m128 r24;
		float r2[4];
	};
	union {
		__m128 r34;
		float r3[4];
	};

#if 0
	for (int i = 0; i < 4; i++)
	{
		r0[i] = rng->Rand();
		r1[i] = rng->Rand();
		r2[i] = rng->Rand();
		r3[i] = rng->Rand();
	}
#else
	memset(r0, 0, sizeof(r0));
	memset(r1, 0, sizeof(r1));
	memset(r2, 0, sizeof(r2));
	memset(r3, 0, sizeof(r3));
#endif

	union {
		__m128 blade4;
		float blade[4];
	};

	blade4 = _mm_castsi128_ps(_mm_castps_si128(_mm_mul_ps(r04, _mm_set1_ps(9.0f))));

	r24 = _mm_sub_ps(r24, _mm_mul_ps(_mm_mul_ps(blade4, _mm_set1_ps(1.0f / 9.0f)), _mm_set1_ps(9.0f)));

	union {
		__m128 x14;
		float x1[4];
	};
	union {
		__m128 y14;
		float y1[4];
	};
	union {
		__m128 x24;
		float x2[4];
	};
	union {
		__m128 y24;
		float y2[4];
	};

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	__m128 piOver4point5_4 = _mm_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m128 bladeParam14 = _mm_mul_ps(blade4, piOver4point5_4);
	x14 = _mm_cos_ps(bladeParam14);
	y14 = _mm_sin_ps(bladeParam14);
	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m128 bladeParam24 = _mm_mul_ps(_mm_add_ps(blade4, one4), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	x24 = _mm_cos_ps(bladeParam24);
	y24 = _mm_sin_ps(bladeParam24);

	// if ((r2 + r3) > 1.0f)
	const int should_store = _mm_movemask_ps(_mm_cmpgt_ps(_mm_add_ps(r24, r34), one4));
	const __m128i mask = _mm_set_epi32((should_store & 1) ? -1 : 0, (should_store & 2) ? -1 : 0, (should_store & 4) ? -1 : 0, (should_store & 8) ? -1 : 0);
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(r2, mask, _mm_sub_ps(one4, r24));
	_mm_maskstore_ps(r3, mask, _mm_sub_ps(one4, r34));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m128 xr4 = _mm_add_ps(_mm_mul_ps(x14, r24), _mm_mul_ps(x24, r34));
	const __m128 yr4 = _mm_add_ps(_mm_mul_ps(y14, r24), _mm_mul_ps(y24, r34));

	union {
		__m128 org_x4;
		float org_x[4];
	};
	union {
		__m128 org_y4;
		float org_y[4];
	};
	union {
		__m128 org_z4;
		float org_z[4];
	};

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m128 lens_size4 = _mm_set1_ps(camera.pos_lensSize.w);
	org_x4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.x),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.x), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.x), yr4))));
	org_y4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.y),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.y), yr4), _mm_mul_ps(_mm_set1_ps(camera.up.y), yr4))));
	org_z4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.z),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.z), yr4), _mm_mul_ps(_mm_set1_ps(camera.up.z), yr4))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m128 u4 = _mm_set_ps(x[0], x[1], x[2], x[3]);
	__m128 v4 = _mm_set_ps(y[0], y[1], y[2], y[3]);

	u4 = _mm_add_ps(u4, r04);
	v4 = _mm_add_ps(v4, r14);

	const __m128 scrwidth4 = _mm_set1_ps(1.0f / float(camera.scrwidth));
	const __m128 scrheight4 = _mm_set1_ps(1.0f / float(camera.scrheight));

	u4 = _mm_mul_ps(u4, scrwidth4);
	v4 = _mm_mul_ps(v4, scrheight4);

	union {
		__m128 pixel_x4;
		float pixel_x[4];
	};
	union {
		__m128 pixel_y4;
		float pixel_y[4];
	};
	union {
		__m128 pixel_z4;
		float pixel_z[4];
	};

	const __m128 p1_x4 = _mm_set1_ps(camera.p1.x);
	const __m128 p1_y4 = _mm_set1_ps(camera.p1.y);
	const __m128 p1_z4 = _mm_set1_ps(camera.p1.z);
	const __m128 right_x4 = _mm_set1_ps(camera.right_spreadAngle.x);
	const __m128 right_y4 = _mm_set1_ps(camera.right_spreadAngle.y);
	const __m128 right_z4 = _mm_set1_ps(camera.right_spreadAngle.z);
	const __m128 up_x4 = _mm_set1_ps(camera.up.x);
	const __m128 up_y4 = _mm_set1_ps(camera.up.y);
	const __m128 up_z4 = _mm_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	pixel_x4 = _mm_add_ps(p1_x4, _mm_add_ps(_mm_mul_ps(u4, right_x4), _mm_mul_ps(v4, up_x4)));
	pixel_y4 = _mm_add_ps(p1_y4, _mm_add_ps(_mm_mul_ps(u4, right_y4), _mm_mul_ps(v4, up_y4)));
	pixel_z4 = _mm_add_ps(p1_z4, _mm_add_ps(_mm_mul_ps(u4, right_z4), _mm_mul_ps(v4, up_z4)));

	union {
		__m128 dir_x4;
		float dir_x[4];
	};
	union {
		__m128 dir_y4;
		float dir_y[4];
	};
	union {
		__m128 dir_z4;
		float dir_z[4];
	};

	dir_x4 = _mm_sub_ps(pixel_x4, org_x4);
	dir_y4 = _mm_sub_ps(pixel_y4, org_y4);
	dir_z4 = _mm_sub_ps(pixel_z4, org_z4);

	__m128 length_squared_4 = _mm_mul_ps(dir_x4, dir_x4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_y4, dir_y4), length_squared_4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_z4, dir_z4), length_squared_4);

	const __m128 inv_length = _mm_div_ps(one4, _mm_sqrt_ps(length_squared_4));
	dir_x4 = _mm_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm_mul_ps(dir_z4, inv_length);

	memcpy(query.ray.org_x, org_x, 4 * sizeof(float));
	memcpy(query.ray.org_y, org_y, 4 * sizeof(float));
	memcpy(query.ray.org_z, org_z, 4 * sizeof(float));

	memcpy(query.ray.dir_x, dir_x, 4 * sizeof(float));
	memcpy(query.ray.dir_y, dir_y, 4 * sizeof(float));
	memcpy(query.ray.dir_z, dir_z, 4 * sizeof(float));

	return query;
}

RTCRayHit8 Ray::GenerateRay8(const CameraParams &camera, std::array<std::pair<int, int>, 8> pixels, rfw::utils::RandomGenerator *rng)
{
	RTCRayHit8 query;

	// TODO

	return query;
}

RTCRayHit16 Ray::GenerateRay16(const CameraParams &camera, std::array<std::pair<int, int>, 16> pixels, rfw::utils::RandomGenerator *rng)
{
	RTCRayHit16 query;

	// TODO

	return query;
}