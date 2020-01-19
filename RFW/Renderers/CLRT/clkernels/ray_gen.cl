#include "random.cl"
#include "shared.h"

float blueNoiseSampler(global uint *blueNoise, int x, int y, int sampleIdx, int sampleDimension);

void GenerateEyeRay(global uint *blueNoise, float3 *O, float3 *D, global CLCamera *camera, const uint pixelIdx,
					const uint sampleIdx, uint *seed);

kernel void generate_rays(uint pathCount, global CLCamera *camera, global uint *blueNoise, global float4 *origins,
						  global float4 *directions)
{
	const uint pathIdx = (uint)get_global_id(0);
	if (pathIdx >= pathCount)
		return;

	const uint samplesTaken = camera->samplesTaken;

	float3 O, D;
	const int scrwidth = camera->scrwidth;
	const int scrheight = camera->scrheight;
	const uint pixelIdx = pathIdx % (scrwidth * scrheight);
	const uint sampleIdx = pathIdx / (scrwidth * scrheight) + samplesTaken;
	uint seed = 0;
	GenerateEyeRay(blueNoise, &O, &D, camera, pixelIdx, sampleIdx, &seed);

	origins[pathIdx] = (float4)(O, as_float(((pathIdx << 8) | 1 /* Camera rays are specular */)));
	directions[pathIdx] = (float4)(D, 0.0);
}

void GenerateEyeRay(global uint *blueNoise, float3 *O, float3 *D, global CLCamera *camera, const uint pixelIdx,
					const uint sampleIdx, uint *seed)
{
	const int sx = (int)pixelIdx % camera->scrwidth;
	const int sy = (int)pixelIdx / camera->scrwidth;

	float r0, r1, r2, r3;
	if (sampleIdx < 256)
	{
		r0 = blueNoiseSampler(blueNoise, sx, sy, (int)sampleIdx, 0);
		r1 = blueNoiseSampler(blueNoise, sx, sy, (int)sampleIdx, 1);
		r2 = blueNoiseSampler(blueNoise, sx, sy, (int)sampleIdx, 2);
		r3 = blueNoiseSampler(blueNoise, sx, sy, (int)sampleIdx, 3);
	}
	else
	{
		r0 = RandomFloat(seed);
		r1 = RandomFloat(seed);
		r2 = RandomFloat(seed);
		r3 = RandomFloat(seed);
	}

	const float blade = (float)(int)(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	float piOver4point5 = 3.14159265359f / 4.5f;

	x1 = cos(blade * piOver4point5);
	y1 = sin(blade * piOver4point5);
	x2 = cos((blade + 1.0f) * piOver4point5);
	y2 = sin((blade + 1.0f) * piOver4point5);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;
	const float3 p1 = camera->p1.xyz;
	const float3 right = camera->right_spreadAngle.xyz;
	const float3 up = camera->up.xyz;

	(*O) = camera->pos_lensSize.xyz + camera->pos_lensSize.w * (right * xr + up * yr);
	const float u = (((float)sx) + r0) * (1.0f / (float)camera->scrwidth);
	const float v = (((float)sy) + r1) * (1.0f / (float)camera->scrheight);
	const float3 pointOnPixel = p1 + u * right + v * up;
	(*D) = normalize(pointOnPixel - (*O));
}
