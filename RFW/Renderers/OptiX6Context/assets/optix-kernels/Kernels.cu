#include <optix_device.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

struct PotentialContribution
{
	float4 Origin;
	float4 Direction;
	float4 Emission;
};

// Provided by optix
rtDeclareVariable(uint3, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, stride, , );

// Must be set by context
rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(uint, pathLength, , );
rtDeclareVariable(uint, sampleIndex, , );
rtDeclareVariable(uint, launch_dim, rtLaunchDim, );

rtDeclareVariable(float4, payload, rtPayload, ); // Primary/secondary ray payload
rtDeclareVariable(uint, visible, rtPayload, );	 // Shadow ray payload
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(float4, hit_data, attribute hit_data, );

rtDeclareVariable(uint, instanceIdx, , );
// Triangle API data
rtDeclareVariable(float2, barycentrics, attribute barycentrics, );

// Path tracing buffers
rtBuffer<float4> accumulator;
rtBuffer<float4> pathStates;
rtBuffer<float4> pathOrigins;
rtBuffer<float4> pathDirections;
rtBuffer<PotentialContribution> connectData;
rtBuffer<uint> blueNoise;

// camera parameters
rtDeclareVariable(float4, posLensSize, , );
rtDeclareVariable(float3, right, , );
rtDeclareVariable(float3, up, , );
rtDeclareVariable(float3, p1, , );
rtDeclareVariable(float, geometryEpsilon, , );
rtDeclareVariable(int3, scrsize, , );

__device__ inline uint WangHash(uint s)
{
	s = ((s ^ 61) ^ (s >> 16)) * 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d, s = s ^ (s >> 15);
	return s;
}
__device__ inline uint RandomInt(uint &s)
{
	s ^= s << 13, s ^= s >> 17, s ^= s << 5;
	return s;
}
__device__ inline float RandomFloat(uint &s) { return RandomInt(s) * 2.3283064365387e-10f; }

__device__ inline float blueNoiseSampler(int x, int y, int sampleIdx, int sampleDimension)
{
	// wrap arguments
	x &= 127;
	y &= 127;
	sampleIdx &= 255;
	sampleDimension &= 255;

	// xor index based on optimized ranking
	const int rankedSampleIndex = sampleIdx ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3];

	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];

	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];

	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

RT_PROGRAM void generatePrimaryRay()
{
	const uint pathIdx = (launch_index.x + launch_index.y * scrsize.x) % (scrsize.x * scrsize.y);
	const uint bufferIndex = (pathLength % 2);
	uint seed = WangHash(pathIdx * 16789 + sampleIndex * 1791);

	const int sx = pathIdx % scrsize.x;
	const int sy = pathIdx / scrsize.x;

#if 1
	const float r0 = blueNoiseSampler(sx, sy, int(sampleIndex), 0);
	const float r1 = blueNoiseSampler(sx, sy, int(sampleIndex), 1);
	float r2 = blueNoiseSampler(sx, sy, int(sampleIndex), 2);
	float r3 = blueNoiseSampler(sx, sy, int(sampleIndex), 3);
#else
	const float r0 = RandomFloat(seed);
	const float r1 = RandomFloat(seed);
	float r2 = RandomFloat(seed);
	float r3 = RandomFloat(seed);
#endif
	const float blade = static_cast<int>(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;

	__sincosf(blade * piOver4point5, &x1, &y1);
	__sincosf((blade + 1.0f) * piOver4point5, &x2, &y2);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;
	const float3 origin = make_float3(posLensSize) + posLensSize.w * (right * xr + up * yr);

	const float u = (static_cast<float>(sx) + r0) * (1.0f / scrsize.x);
	const float v = (static_cast<float>(sy) + r1) * (1.0f / scrsize.y);
	const float3 pointOnPixel = p1 + u * right + v * up;
	const float3 direction = normalize(pointOnPixel - origin);

	const uint bufferIdx = pathIdx + (bufferIndex * stride);

	pathOrigins[bufferIdx] = make_float4(origin, __uint_as_float((pathIdx << 8) + 1 /* 1 == specular */));
	pathDirections[bufferIdx] = make_float4(direction, 0);

	float4 result = make_float4(0, 0, __int_as_float(-1), 0);

	rtTrace(sceneRoot, make_Ray(origin, direction, 0u, 10.0f * geometryEpsilon, RT_DEFAULT_MAX), result);
	pathStates[bufferIdx] = result;
}

RT_PROGRAM void generateSecondaryRay()
{
	const uint pathIdx = (launch_index.x + launch_index.y * scrsize.x) % (scrsize.x * scrsize.y);
	const uint bufferIndex = (pathLength % 2);
	const uint bufferIdx = pathIdx + (bufferIndex * stride);

	const float4 O4 = pathOrigins[bufferIdx];
	const float4 D4 = pathDirections[bufferIdx];

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);

	float4 result = make_float4(0, 0, __int_as_float(-1), 0);
	rtTrace(sceneRoot, make_Ray(O, D, 1u, 10.0f * geometryEpsilon, RT_DEFAULT_MAX), result);
	pathStates[bufferIdx] = result;
}

RT_PROGRAM void generateShadowRay()
{
	const uint pathIdx = (launch_index.x + launch_index.y * scrsize.x) % (scrsize.x * scrsize.y);
	const float4 O4 = connectData[pathIdx].Origin;
	const float4 D4 = connectData[pathIdx].Direction;

	const float3 O = make_float3(O4);
	const float3 D = make_float3(D4);

	uint isVisible = 0;
	const auto epsilon = 10.0f * geometryEpsilon;
	rtTrace(sceneRoot, make_Ray(O, D, 2u, 1e-4f, D4.w - 2e-5f), isVisible);
	if (isVisible == 0)
		return;

	const float4 contribution = connectData[pathIdx].Emission;
	const uint pixelIdx = __float_as_uint(contribution.w);
	accumulator[pixelIdx] += make_float4(make_float3(contribution), 1.0f);
}

RT_PROGRAM void closestHit() { payload = hit_data; }
RT_PROGRAM void exception()
{
	rtPrintf("Caught exception %i at launch index (%d)\n", rtGetExceptionCode(), launch_index);
	rtPrintExceptionDetails();
}
RT_PROGRAM void missShadow() { visible = 1; }
RT_PROGRAM void triangleAttributes()
{
	const float2 bary = rtGetTriangleBarycentrics();
	const uint primIdx = rtGetPrimitiveIndex();
	const uint barycentrics = uint(65535.0f * bary.x) + (uint(65535.0f * bary.y) << 16);

	hit_data = make_float4(__uint_as_float(barycentrics), __uint_as_float(instanceIdx), __int_as_float(primIdx), t_hit);
}