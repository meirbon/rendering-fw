#include "random.cl"
#include "shared.h"

float blueNoiseSampler(global uint *blueNoise, int x, int y, int sampleIdx, int sampleDimension);

void GenerateEyeRay(global uint *blueNoise, float3 *O, float3 *D, global CLCamera *camera, const uint pixelIdx,
					const uint sampleIdx, uint *seed);

kernel void generate_rays(uint pathCount,			 // 0
						  global CLCamera *camera,	 // 1
						  global uint *blueNoise,	 // 2
						  global float4 *origins,	 // 3
						  global float4 *directions, // 4
						  float4 posLensSize,		 // 5
						  float4 p1,				 // 6
						  float4 right_spreadAngle,	 // 7
						  float4 up,				 // 8
						  uint scrwidth,			 // 9
						  uint scrheight			 // 10
)
{
	const uint pathIdx = (uint)get_global_id(0);
	if (pathIdx >= pathCount)
		return;

	const uint samplesTaken = camera->samplesTaken;
	const uint pixelIdx = pathIdx % (scrwidth * scrheight);
	const uint sampleIdx = pathIdx / (scrwidth * scrheight) + samplesTaken;
	uint seed = 0;

	const int sx = (int)pixelIdx % scrwidth;
	const int sy = (int)pixelIdx / scrwidth;

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
		r0 = RandomFloat(&seed);
		r1 = RandomFloat(&seed);
		r2 = RandomFloat(&seed);
		r3 = RandomFloat(&seed);
	}

	const float blade = round(r0 * 9.0f);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float piOver4point5 = 3.14159265359f / 4.5f;

	float y1;
	float x1 = sincos(blade * piOver4point5, &y1);
	float y2;
	float x2 = sincos((blade + 1.0f) * piOver4point5, &y2);

	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}

	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;

	const float3 O = posLensSize.xyz + posLensSize.w * (right_spreadAngle.xyz * xr + up.xyz * yr);
	const float u = (((float)sx) + r0) * (1.0f / (float)scrwidth);
	const float v = (((float)sy) + r1) * (1.0f / (float)scrheight);
	const float3 pointOnPixel = p1.xyz + u * right_spreadAngle.xyz + v * up.xyz;
	const float3 D = normalize(pointOnPixel - O);

	// TODO: Data doesn't seem to get written correctly
	origins[pathIdx] = (float4)(O.xyz, as_float(((pixelIdx << 8) | 1 /* Camera rays are specular */)));
	directions[pathIdx] = (float4)(D.xyz, 0.0);
}
