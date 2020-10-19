#include "PCH.h"

cpurt::Ray::CameraParams::CameraParams(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height)
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

cpurt::Ray cpurt::Ray::generateFromView(const cpurt::Ray::CameraParams &camera, int x, int y, float r0, float r1, float r2, float r3)
{
	cpurt::Ray ray;
	const float blade = float(int(r0 * 9));
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	const float piOver4point5 = 3.14159265359f / 4.5f;
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

cpurt::RayPacket4 cpurt::Ray::generate_ray4(const CameraParams &camera, const int x[4], const int y[4], rfw::utils::rng *rng)
{
	cpurt::RayPacket4 query = {};

	for (int i = 0; i < 4; i++)
		query.pixelID[i] = camera.scrwidth * y[i] + x[i];

	static const __m128 one4 = _mm_set1_ps(1.0f);

	const __m128 r04 = _mm_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand());
	const __m128 r14 = _mm_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand());
	__m128 r24 = _mm_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand());
	__m128 r34 = _mm_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand());

	const __m128 blade4 = _mm_mul_ps(r04, _mm_set1_ps(9.0f));

	r24 = _mm_sub_ps(r24, _mm_mul_ps(_mm_mul_ps(blade4, _mm_set1_ps(1.0f / 9.0f)), _mm_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	static const __m128 piOver4point5_4 = _mm_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m128 bladeParam14 = _mm_mul_ps(blade4, piOver4point5_4);

	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);
	rfw::simd::vector4 x14, y14;
	rfw::simd::sincos(bladeParam14, &x14, &y14);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m128 bladeParam24 = _mm_mul_ps(_mm_add_ps(blade4, one4), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	rfw::simd::vector4 x24, y24;
	rfw::simd::sincos(bladeParam24, &x24, &y24);

	// if ((r2 + r3) > 1.0f)
	const __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_add_ps(r24, r34), one4));
	// r2 = 1.0f - r2;
	// r3 = 1.0f - r3;
	_mm_maskstore_ps(reinterpret_cast<float *>(&r24), mask, _mm_sub_ps(one4, r24));
	_mm_maskstore_ps(reinterpret_cast<float *>(&r34), mask, _mm_sub_ps(one4, r34));

	// const float xr = x1 * r2 + x2 * r3;
	// const float yr = y1 * r2 + y2 * r3;
	const __m128 xr4 = _mm_add_ps(_mm_mul_ps(x14.vec_4, r24), _mm_mul_ps(x24.vec_4, r34));
	const __m128 yr4 = _mm_add_ps(_mm_mul_ps(y14.vec_4, r24), _mm_mul_ps(y24.vec_4, r34));

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
	__m128 u4 = _mm_setr_ps(float(x[0]), float(x[1]), float(x[2]), float(x[3]));
	__m128 v4 = _mm_setr_ps(float(y[0]), float(y[1]), float(y[2]), float(y[3]));

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

	// ray.direction = pointOnPixel - ray.origin;
	__m128 dir_x4 = _mm_sub_ps(pixel_x4, org_x4);
	__m128 dir_y4 = _mm_sub_ps(pixel_y4, org_y4);
	__m128 dir_z4 = _mm_sub_ps(pixel_z4, org_z4);

	// ray.direction = normalize(ray.direction);
	__m128 length_squared_4 = _mm_mul_ps(dir_x4, dir_x4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_y4, dir_y4), length_squared_4);
	length_squared_4 = _mm_add_ps(_mm_mul_ps(dir_z4, dir_z4), length_squared_4);
	const __m128 inv_length = _mm_div_ps(one4, _mm_sqrt_ps(length_squared_4));
	dir_x4 = _mm_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm_mul_ps(dir_z4, inv_length);

	memcpy(query.origin_x, &org_x4, 4 * sizeof(float));
	memcpy(query.origin_y, &org_y4, 4 * sizeof(float));
	memcpy(query.origin_z, &org_z4, 4 * sizeof(float));

	memcpy(query.direction_x, &dir_x4, 4 * sizeof(float));
	memcpy(query.direction_y, &dir_y4, 4 * sizeof(float));
	memcpy(query.direction_z, &dir_z4, 4 * sizeof(float));

	return query;
}

cpurt::RayPacket8 cpurt::Ray::generate_ray8(const CameraParams &camera, const int x[8], const int y[8], rfw::utils::rng *rng)
{
	cpurt::RayPacket8 query = {};

	for (int i = 0; i < 8; i++)
		query.pixelID[i] = camera.scrwidth * y[i] + x[i];

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

	r04 = _mm256_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand());
	r14 = _mm256_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand());
	r24 = _mm256_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand());
	r34 = _mm256_set_ps(rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand(), rng->rand());

	const __m256 blade4 = _mm256_mul_ps(r04, _mm256_set1_ps(9.0f));

	r24 = _mm256_sub_ps(r24, _mm256_mul_ps(_mm256_mul_ps(blade4, _mm256_set1_ps(1.0f / 9.0f)), _mm256_set1_ps(9.0f)));

	// const float blade = float(int(r0 * 9));
	// r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	// float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;
	const __m256 piOver4point5_4 = _mm256_set1_ps(piOver4point5);
	// float bladeParam = blade * piOver4point5;
	const __m256 bladeParam14 = _mm256_mul_ps(blade4, piOver4point5_4);

	// x1 = cos(bladeParam);
	// y1 = sin(bladeParam);
	union {
		__m256 x14;
		rfw::simd::vector4 x1_4[2];
	};
	union {
		__m256 y14;
		rfw::simd::vector4 y1_4[2];
	};
	rfw::simd::sincos(_mm256_extractf128_ps(bladeParam14, 0), &x1_4[0], &y1_4[0]);
	rfw::simd::sincos(_mm256_extractf128_ps(bladeParam14, 1), &x1_4[1], &y1_4[1]);

	// bladeParam = (blade + 1.0f) * piOver4point5;
	const __m256 bladeParam24 = _mm256_mul_ps(_mm256_add_ps(blade4, one8), piOver4point5_4);
	// x2 = cos(bladeParam);
	// y2 = sin(bladeParam);
	union {
		__m256 x24;
		rfw::simd::vector4 x2_4[2];
	};
	union {
		__m256 y24;
		rfw::simd::vector4 y2_4[2];
	};
	rfw::simd::sincos(_mm256_extractf128_ps(bladeParam24, 0), &x2_4[0], &y2_4[0]);
	rfw::simd::sincos(_mm256_extractf128_ps(bladeParam24, 1), &x2_4[1], &y2_4[1]);

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
	__m256 u4 = _mm256_setr_ps(float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5]), float(x[6]), float(x[7]));
	__m256 v4 = _mm256_setr_ps(float(y[0]), float(y[1]), float(y[2]), float(y[3]), float(y[4]), float(y[5]), float(y[6]), float(y[7]));

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

	__m256 length_squared_4 = _mm256_mul_ps(dir_x4, dir_x4);
	length_squared_4 = _mm256_add_ps(_mm256_mul_ps(dir_y4, dir_y4), length_squared_4);
	length_squared_4 = _mm256_add_ps(_mm256_mul_ps(dir_z4, dir_z4), length_squared_4);

	const __m256 inv_length = _mm256_div_ps(one8, _mm256_sqrt_ps(length_squared_4));
	dir_x4 = _mm256_mul_ps(dir_x4, inv_length);
	dir_y4 = _mm256_mul_ps(dir_y4, inv_length);
	dir_z4 = _mm256_mul_ps(dir_z4, inv_length);

	_mm256_store_ps(query.origin_x, org_x4);
	_mm256_store_ps(query.origin_y, org_y4);
	_mm256_store_ps(query.origin_z, org_z4);

	_mm256_store_ps(query.direction_x, dir_x4);
	_mm256_store_ps(query.direction_y, dir_y4);
	_mm256_store_ps(query.direction_z, dir_z4);

	return query;
}
