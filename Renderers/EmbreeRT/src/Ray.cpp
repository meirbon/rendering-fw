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
	RTCRayHit4 query{};

	for (int i = 0; i < 4; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	static const __m128 one4 = _mm_set1_ps(1.0f);

	const __m128 r04 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	const __m128 r14 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	__m128 r24 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	__m128 r34 = _mm_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());

	const __m128 blade4 = _mm_mul_ps(r04, _mm_set1_ps(9.0f));

	r24 = _mm_sub_ps(r24, _mm_mul_ps(_mm_mul_ps(blade4, _mm_set1_ps(1.0f / 9.0f)), _mm_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	static const __m128 piOver4point5_4 = _mm_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m128 bladeParam14 = _mm_mul_ps(blade4, piOver4point5_4);
	const __m128 x14 = _mm_cos_ps(bladeParam14);
	const __m128 y14 = _mm_sin_ps(bladeParam14);
	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m128 bladeParam24 = _mm_mul_ps(_mm_add_ps(blade4, one4), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	const __m128 x24 = _mm_cos_ps(bladeParam24);
	const __m128 y24 = _mm_sin_ps(bladeParam24);

	// if ((r2 + r3) > 1.0f)
	const __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(r24, r34), one4));
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(reinterpret_cast<float *>(&r24), mask, _mm_sub_ps(one4, r24));
	_mm_maskstore_ps(reinterpret_cast<float *>(&r34), mask, _mm_sub_ps(one4, r34));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m128 xr4 = _mm_add_ps(_mm_mul_ps(x14, r24), _mm_mul_ps(x24, r34));
	const __m128 yr4 = _mm_add_ps(_mm_mul_ps(y14, r24), _mm_mul_ps(y24, r34));

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m128 lens_size4 = _mm_set1_ps(camera.pos_lensSize.w);
	const __m128 org_x4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.x),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.x), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.x), yr4))));
	const __m128 org_y4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.y),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.y), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.y), yr4))));
	const __m128 org_z4 =
		_mm_add_ps(_mm_set1_ps(camera.pos_lensSize.z),
				   _mm_mul_ps(lens_size4, _mm_add_ps(_mm_mul_ps(_mm_set1_ps(camera.right_spreadAngle.z), xr4), _mm_mul_ps(_mm_set1_ps(camera.up.z), yr4))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m128 u4 = _mm_setr_ps(x[0], x[1], x[2], x[3]);
	__m128 v4 = _mm_setr_ps(y[0], y[1], y[2], y[3]);

	u4 = _mm_add_ps(u4, r04);
	v4 = _mm_add_ps(v4, r14);

	const __m128 scrwidth4 = _mm_set1_ps(1.0f / float(camera.scrwidth));
	const __m128 scrheight4 = _mm_set1_ps(1.0f / float(camera.scrheight));

	u4 = _mm_mul_ps(u4, scrwidth4);
	v4 = _mm_mul_ps(v4, scrheight4);

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
	const __m128 pixel_x4 = _mm_add_ps(p1_x4, _mm_add_ps(_mm_mul_ps(u4, right_x4), _mm_mul_ps(v4, up_x4)));
	const __m128 pixel_y4 = _mm_add_ps(p1_y4, _mm_add_ps(_mm_mul_ps(u4, right_y4), _mm_mul_ps(v4, up_y4)));
	const __m128 pixel_z4 = _mm_add_ps(p1_z4, _mm_add_ps(_mm_mul_ps(u4, right_z4), _mm_mul_ps(v4, up_z4)));

	__m128 dir_x4 = _mm_sub_ps(pixel_x4, org_x4);
	__m128 dir_y4 = _mm_sub_ps(pixel_y4, org_y4);
	__m128 dir_z4 = _mm_sub_ps(pixel_z4, org_z4);

	__m128 length_squared_4 = _mm_dp_ps(dir_x4, dir_x4, 0xFF);
	length_squared_4 = _mm_add_ps(_mm_dp_ps(dir_y4, dir_y4, 0xFF), length_squared_4);
	length_squared_4 = _mm_add_ps(_mm_dp_ps(dir_z4, dir_z4, 0xFF), length_squared_4);

	const __m128 inv_length = _mm_div_ps(one4, _mm_sqrt_ps(length_squared_4));
	dir_x4 = _mm_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm_mul_ps(dir_z4, inv_length);

	memcpy(query.ray.org_x, &org_x4, 4 * sizeof(float));
	memcpy(query.ray.org_y, &org_y4, 4 * sizeof(float));
	memcpy(query.ray.org_z, &org_z4, 4 * sizeof(float));

	memcpy(query.ray.dir_x, &dir_x4, 4 * sizeof(float));
	memcpy(query.ray.dir_y, &dir_y4, 4 * sizeof(float));
	memcpy(query.ray.dir_z, &dir_z4, 4 * sizeof(float));

	return query;
}

RTCRayHit8 Ray::GenerateRay8(const CameraParams &camera, const int x[8], const int y[8], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit8 query{};

	for (int i = 0; i < 8; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	const __m256 one8 = _mm256_set1_ps(1.0f);

	union {
		__m256 r04;
		float r0[8];
	};
	union {
		__m256 r14;
		float r1[8];
	};
	union {
		__m256 r24;
		float r2[8];
	};
	union {
		__m256 r34;
		float r3[8];
	};

	r04 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r14 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r24 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());
	r34 = _mm256_set_ps(rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand(), rng->Rand());

	const __m256 blade4 = _mm256_mul_ps(r04, _mm256_set1_ps(9.0f));

	r24 = _mm256_sub_ps(r24, _mm256_mul_ps(_mm256_mul_ps(blade4, _mm256_set1_ps(1.0f / 9.0f)), _mm256_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	const __m256 piOver4point5_4 = _mm256_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m256 bladeParam14 = _mm256_mul_ps(blade4, piOver4point5_4);
	__m256 x14 = _mm256_cos_ps(bladeParam14);
	__m256 y14 = _mm256_sin_ps(bladeParam14);
	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m256 bladeParam24 = _mm256_mul_ps(_mm256_add_ps(blade4, one8), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	__m256 x24 = _mm256_cos_ps(bladeParam24);
	__m256 y24 = _mm256_sin_ps(bladeParam24);

	// if ((r2 + r3) > 1.0f)
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128i mask1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r24, 0), _mm256_extractf128_ps(r34, 0)), one4));
	const __m128i mask2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r24, 1), _mm256_extractf128_ps(r34, 1)), one4));
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(r2, mask1, _mm_sub_ps(one4, _mm256_extractf128_ps(r24, 0)));
	_mm_maskstore_ps(r3, mask2, _mm_sub_ps(one4, _mm256_extractf128_ps(r34, 0)));
	_mm_maskstore_ps(r2 + 4, mask1, _mm_sub_ps(one4, _mm256_extractf128_ps(r24, 1)));
	_mm_maskstore_ps(r3 + 4, mask2, _mm_sub_ps(one4, _mm256_extractf128_ps(r34, 1)));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m256 xr4 = _mm256_add_ps(_mm256_mul_ps(x14, r24), _mm256_mul_ps(x24, r34));
	const __m256 yr4 = _mm256_add_ps(_mm256_mul_ps(y14, r24), _mm256_mul_ps(y24, r34));

	union {
		__m256 org_x4;
		float org_x[8];
	};
	union {
		__m256 org_y4;
		float org_y[8];
	};
	union {
		__m256 org_z4;
		float org_z[8];
	};

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m256 lens_size4 = _mm256_set1_ps(camera.pos_lensSize.w);
	org_x4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr4))));
	org_y4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), yr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr4))));
	org_z4 = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
						   _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), yr4),
																   _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr4))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m256 u4 = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
	__m256 v4 = _mm256_setr_ps(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);

	u4 = _mm256_add_ps(u4, r04);
	v4 = _mm256_add_ps(v4, r14);

	const __m256 scrwidth4 = _mm256_set1_ps(1.0f / float(camera.scrwidth));
	const __m256 scrheight4 = _mm256_set1_ps(1.0f / float(camera.scrheight));

	u4 = _mm256_mul_ps(u4, scrwidth4);
	v4 = _mm256_mul_ps(v4, scrheight4);

	union {
		__m256 pixel_x4;
		float pixel_x[8];
	};
	union {
		__m256 pixel_y4;
		float pixel_y[8];
	};
	union {
		__m256 pixel_z4;
		float pixel_z[8];
	};

	const __m256 p1_x4 = _mm256_set1_ps(camera.p1.x);
	const __m256 p1_y4 = _mm256_set1_ps(camera.p1.y);
	const __m256 p1_z4 = _mm256_set1_ps(camera.p1.z);
	const __m256 right_x4 = _mm256_set1_ps(camera.right_spreadAngle.x);
	const __m256 right_y4 = _mm256_set1_ps(camera.right_spreadAngle.y);
	const __m256 right_z4 = _mm256_set1_ps(camera.right_spreadAngle.z);
	const __m256 up_x4 = _mm256_set1_ps(camera.up.x);
	const __m256 up_y4 = _mm256_set1_ps(camera.up.y);
	const __m256 up_z4 = _mm256_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	pixel_x4 = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u4, right_x4), _mm256_mul_ps(v4, up_x4)));
	pixel_y4 = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u4, right_y4), _mm256_mul_ps(v4, up_y4)));
	pixel_z4 = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u4, right_z4), _mm256_mul_ps(v4, up_z4)));

	__m256 dir_x4;
	__m256 dir_y4;
	__m256 dir_z4;

	dir_x4 = _mm256_sub_ps(pixel_x4, org_x4);
	dir_y4 = _mm256_sub_ps(pixel_y4, org_y4);
	dir_z4 = _mm256_sub_ps(pixel_z4, org_z4);

	__m256 length_squared_4 = _mm256_dp_ps(dir_x4, dir_x4, 0xFF);
	length_squared_4 = _mm256_add_ps(_mm256_dp_ps(dir_y4, dir_y4, 0xFF), length_squared_4);
	length_squared_4 = _mm256_add_ps(_mm256_dp_ps(dir_z4, dir_z4, 0xFF), length_squared_4);

	const __m256 inv_length = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4));
	dir_x4 = _mm256_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm256_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm256_mul_ps(dir_z4, inv_length);

	_mm256_store_ps(query.ray.org_x, org_x4);
	_mm256_store_ps(query.ray.org_y, org_y4);
	_mm256_store_ps(query.ray.org_z, org_z4);

	_mm256_store_ps(query.ray.dir_x, dir_x4);
	_mm256_store_ps(query.ray.dir_y, dir_y4);
	_mm256_store_ps(query.ray.dir_z, dir_z4);

	return query;
}

RTCRayHit16 Ray::GenerateRay16(const CameraParams &camera, const int x[16], const int y[16], rfw::utils::RandomGenerator *rng)
{
	RTCRayHit16 query{};

	for (int i = 0; i < 16; i++)
	{
		query.ray.tfar[i] = 1e34f;
		query.ray.tnear[i] = 1e-5f;
		query.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
		query.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
		query.ray.id[i] = camera.scrwidth * y[i] + x[i];
	}

	const __m256 one8 = _mm256_set1_ps(1.0f);

	union {
		__m256 r0_16[2];
		float r0[16];
	};
	union {
		__m256 r1_16[2];
		float r1[16];
	};
	union {
		__m256 r2_16[2];
		float r2[16];
	};
	union {
		__m256 r3_16[2];
		float r3[16];
	};

	for (int i = 0; i < 16; i++)
	{
		r0[i] = rng->Rand();
		r1[i] = rng->Rand();
		r2[i] = rng->Rand();
		r3[i] = rng->Rand();
	}

	const __m256 blade0_8 = _mm256_mul_ps(r0_16[0], _mm256_set1_ps(9.0f));
	const __m256 blade1_8 = _mm256_mul_ps(r0_16[1], _mm256_set1_ps(9.0f));
	static const __m256 one_over_9 = _mm256_set1_ps(1.0f / 9.0f);
	static const __m256 nine = _mm256_set1_ps(9.0f);

	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	r2_16[0] = _mm256_sub_ps(r2_16[0], _mm256_mul_ps(_mm256_mul_ps(blade0_8, one_over_9), nine));
	r2_16[1] = _mm256_sub_ps(r2_16[1], _mm256_mul_ps(_mm256_mul_ps(blade1_8, one_over_9), nine));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	static const __m256 piOver4point5_4 = _mm256_set1_ps(3.14159265359f / 4.5f);
	// float bladeParam = blade * piOver4point5;
	const __m256 bladeParam1_0 = _mm256_mul_ps(blade0_8, piOver4point5_4);
	const __m256 bladeParam1_1 = _mm256_mul_ps(blade1_8, piOver4point5_4);

	__m256 x1_8[2];
	__m256 y1_8[2];
	__m256 x2_8[2];
	__m256 y2_8[2];

	x1_8[0] = _mm256_cos_ps(bladeParam1_0);
	y1_8[0] = _mm256_sin_ps(bladeParam1_0);
	x1_8[1] = _mm256_cos_ps(bladeParam1_1);
	y1_8[1] = _mm256_sin_ps(bladeParam1_1);
	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m256 bladeParam2_0 = _mm256_mul_ps(_mm256_add_ps(blade0_8, one8), piOver4point5_4);
	const __m256 bladeParam2_1 = _mm256_mul_ps(_mm256_add_ps(blade1_8, one8), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	x2_8[0] = _mm256_cos_ps(bladeParam2_0);
	y2_8[0] = _mm256_sin_ps(bladeParam2_0);
	x2_8[1] = _mm256_cos_ps(bladeParam2_1);
	y2_8[1] = _mm256_sin_ps(bladeParam2_1);

	// if ((r2 + r3) > 1.0f)
	const __m128 one4 = _mm_set1_ps(1.0f);
	const __m128i mask0_1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[0], 0), _mm256_extractf128_ps(r3_16[0], 0)), one4));
	const __m128i mask0_2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[0], 1), _mm256_extractf128_ps(r3_16[0], 1)), one4));
	const __m128i mask1_1 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[1], 0), _mm256_extractf128_ps(r3_16[1], 0)), one4));
	const __m128i mask1_2 = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(_mm256_extractf128_ps(r2_16[1], 1), _mm256_extractf128_ps(r3_16[1], 1)), one4));

	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(r2, mask0_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[0], 0)));
	_mm_maskstore_ps(r3, mask0_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[0], 0)));
	_mm_maskstore_ps(r2 + 4, mask0_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[0], 1)));
	_mm_maskstore_ps(r3 + 4, mask0_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[0], 1)));

	_mm_maskstore_ps(r2 + 8, mask1_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[1], 0)));
	_mm_maskstore_ps(r3 + 8, mask1_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[1], 0)));
	_mm_maskstore_ps(r2 + 12, mask1_1, _mm_sub_ps(one4, _mm256_extractf128_ps(r2_16[1], 1)));
	_mm_maskstore_ps(r3 + 12, mask1_2, _mm_sub_ps(one4, _mm256_extractf128_ps(r3_16[1], 1)));

	__m256 xr_8[2];
	__m256 yr_8[2];

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	xr_8[0] = _mm256_add_ps(_mm256_mul_ps(x1_8[0], r2_16[0]), _mm256_mul_ps(x2_8[0], r3_16[0]));
	yr_8[0] = _mm256_add_ps(_mm256_mul_ps(y1_8[0], r2_16[0]), _mm256_mul_ps(y2_8[0], r3_16[0]));
	xr_8[1] = _mm256_add_ps(_mm256_mul_ps(x1_8[1], r2_16[1]), _mm256_mul_ps(x2_8[1], r3_16[1]));
	yr_8[1] = _mm256_add_ps(_mm256_mul_ps(y1_8[1], r2_16[1]), _mm256_mul_ps(y2_8[1], r3_16[1]));

	union {
		__m256 org_x4[2];
		float org_x[16];
	};
	union {
		__m256 org_y4[2];
		float org_y[16];
	};
	union {
		__m256 org_z4[2];
		float org_z[16];
	};

	// ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const __m256 lens_size4 = _mm256_set1_ps(camera.pos_lensSize.w);

	org_x4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr_8[0]))));
	org_x4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.x),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.x), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.x), yr_8[1]))));
	org_y4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr_8[0]))));
	org_y4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.y),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.y), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.y), yr_8[1]))));
	org_z4[0] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), xr_8[0]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr_8[0]))));
	org_z4[1] = _mm256_add_ps(_mm256_set1_ps(camera.pos_lensSize.z),
							  _mm256_mul_ps(lens_size4, _mm256_add_ps(_mm256_mul_ps(_mm256_set1_ps(camera.right_spreadAngle.z), xr_8[1]),
																	  _mm256_mul_ps(_mm256_set1_ps(camera.up.z), yr_8[1]))));

	// const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	// const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	__m256 u8[2];
	u8[0] = _mm256_setr_ps(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]);
	u8[1] = _mm256_setr_ps(x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15]);
	__m256 v8[2];
	v8[0] = _mm256_setr_ps(y[0], y[1], y[2], y[3], y[4], y[5], y[6], y[7]);
	v8[1] = _mm256_setr_ps(y[8], y[9], y[10], y[11], y[12], y[13], y[14], y[15]);

	u8[0] = _mm256_add_ps(u8[0], r0_16[0]);
	u8[1] = _mm256_add_ps(u8[1], r0_16[1]);
	v8[0] = _mm256_add_ps(v8[0], r1_16[0]);
	v8[1] = _mm256_add_ps(v8[1], r1_16[1]);

	const __m256 scrwidth4 = _mm256_set1_ps(1.0f / float(camera.scrwidth));
	const __m256 scrheight4 = _mm256_set1_ps(1.0f / float(camera.scrheight));

	u8[0] = _mm256_mul_ps(u8[0], scrwidth4);
	u8[1] = _mm256_mul_ps(u8[1], scrwidth4);
	v8[0] = _mm256_mul_ps(v8[0], scrheight4);
	v8[1] = _mm256_mul_ps(v8[1], scrheight4);

	__m256 pixel_x4[2];
	__m256 pixel_y4[2];
	__m256 pixel_z4[2];

	const __m256 p1_x4 = _mm256_set1_ps(camera.p1.x);
	const __m256 p1_y4 = _mm256_set1_ps(camera.p1.y);
	const __m256 p1_z4 = _mm256_set1_ps(camera.p1.z);
	const __m256 right_x4 = _mm256_set1_ps(camera.right_spreadAngle.x);
	const __m256 right_y4 = _mm256_set1_ps(camera.right_spreadAngle.y);
	const __m256 right_z4 = _mm256_set1_ps(camera.right_spreadAngle.z);
	const __m256 up_x4 = _mm256_set1_ps(camera.up.x);
	const __m256 up_y4 = _mm256_set1_ps(camera.up.y);
	const __m256 up_z4 = _mm256_set1_ps(camera.up.z);

	// const vec3 pointOnPixel = p1 + u * right + v * up;
	pixel_x4[0] = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_x4), _mm256_mul_ps(v8[0], up_x4)));
	pixel_x4[1] = _mm256_add_ps(p1_x4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_x4), _mm256_mul_ps(v8[1], up_x4)));
	pixel_y4[0] = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_y4), _mm256_mul_ps(v8[0], up_y4)));
	pixel_y4[1] = _mm256_add_ps(p1_y4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_y4), _mm256_mul_ps(v8[1], up_y4)));
	pixel_z4[0] = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u8[0], right_z4), _mm256_mul_ps(v8[0], up_z4)));
	pixel_z4[1] = _mm256_add_ps(p1_z4, _mm256_add_ps(_mm256_mul_ps(u8[1], right_z4), _mm256_mul_ps(v8[1], up_z4)));

	union {
		__m256 dir_x4[2];
		float dir_x[16];
	};
	union {
		__m256 dir_y4[2];
		float dir_y[16];
	};
	union {
		__m256 dir_z4[2];
		float dir_z[16];
	};

	dir_x4[0] = _mm256_sub_ps(pixel_x4[0], org_x4[0]);
	dir_x4[1] = _mm256_sub_ps(pixel_x4[1], org_x4[1]);
	dir_y4[0] = _mm256_sub_ps(pixel_y4[0], org_y4[0]);
	dir_y4[1] = _mm256_sub_ps(pixel_y4[1], org_y4[1]);
	dir_z4[0] = _mm256_sub_ps(pixel_z4[0], org_z4[0]);
	dir_z4[1] = _mm256_sub_ps(pixel_z4[1], org_z4[1]);

	__m256 length_squared_4[2];
	length_squared_4[0] = _mm256_mul_ps(dir_x4[0], dir_x4[0]);
	length_squared_4[0] = _mm256_add_ps(_mm256_mul_ps(dir_y4[0], dir_y4[0]), length_squared_4[0]);
	length_squared_4[0] = _mm256_add_ps(_mm256_mul_ps(dir_z4[0], dir_z4[0]), length_squared_4[0]);

	length_squared_4[1] = _mm256_mul_ps(dir_x4[1], dir_x4[1]);
	length_squared_4[1] = _mm256_add_ps(_mm256_mul_ps(dir_y4[1], dir_y4[1]), length_squared_4[1]);
	length_squared_4[1] = _mm256_add_ps(_mm256_mul_ps(dir_z4[1], dir_z4[1]), length_squared_4[1]);

	__m256 inv_length[2];
	inv_length[0] = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4[0]));
	inv_length[1] = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4[1]));

	dir_x4[0] = _mm256_mul_ps(dir_x4[0], inv_length[0]);
	dir_x4[1] = _mm256_mul_ps(dir_x4[1], inv_length[1]);
	dir_y4[0] = _mm256_mul_ps(dir_y4[0], inv_length[0]);
	dir_y4[1] = _mm256_mul_ps(dir_y4[1], inv_length[1]);
	dir_z4[0] = _mm256_mul_ps(dir_z4[0], inv_length[0]);
	dir_z4[1] = _mm256_mul_ps(dir_z4[1], inv_length[1]);

	memcpy(query.ray.org_x, org_x, 16 * sizeof(float));
	memcpy(query.ray.org_y, org_y, 16 * sizeof(float));
	memcpy(query.ray.org_z, org_z, 16 * sizeof(float));

	memcpy(query.ray.dir_x, dir_x, 16 * sizeof(float));
	memcpy(query.ray.dir_y, dir_y, 16 * sizeof(float));
	memcpy(query.ray.dir_z, dir_z, 16 * sizeof(float));

	return query;
}