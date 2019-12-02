#define GLM_FORCE_ALIGNED_GENTYPES

#include <optix.h>
#include <optix_device.h>
#include <optix_world.h>
#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>
#include <optixu/optixu_aabb_namespace.h>
#include <SharedStructs.h>
#include <Settings.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace optix;
using namespace glm;

// Provided by optix
rtDeclareVariable(uint, launch_index, rtLaunchIndex, );
rtDeclareVariable(uint, stride, , );

// Must be set by context
rtDeclareVariable(rtObject, sceneRoot, , );
rtDeclareVariable(uint, pathLength, , );
rtDeclareVariable(uint, sampleIndex, , );
rtDeclareVariable(uint, launch_dim, rtLaunchDim, );

rtDeclareVariable(glm::vec4, payload, rtPayload, ); // Primary/secondary ray payload
rtDeclareVariable(uint, visible, rtPayload, );		// Shadow ray payload
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(Ray, ray, rtCurrentRay, );
rtDeclareVariable(glm::vec4, hit_data, attribute hit_data, );

rtDeclareVariable(uint, instanceIdx, , );
// Triangle API data
rtDeclareVariable(glm::vec2, barycentrics, attribute barycentrics, );

// Path tracing buffers
rtBuffer<glm::vec4> accumulator;
rtBuffer<glm::vec4> pathStates;
rtBuffer<glm::vec4> pathOrigins;
rtBuffer<glm::vec4> pathDirections;
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
	const uint pathIdx = launch_index % (scrsize.x * scrsize.y);
	const uint bufferIndex = (pathLength % 2);
	uint seed = WangHash(pathIdx * 16789 + sampleIndex * 1791);

	const int sx = pathIdx % scrsize.x;
	const int sy = pathIdx / scrsize.x;

#ifdef BLUENOISE
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
	const glm::vec4 posLens = glm::vec4(posLensSize.x, posLensSize.y, posLensSize.z, posLensSize.w);
	const glm::vec3 origin =
		glm::vec3(posLens) + posLens.w * (glm::vec3(right.x, right.y, right.z) * xr + glm::vec3(up.x, up.y, up.z) * yr);

	const float u = (static_cast<float>(sx) + r0) * (1.0f / scrsize.x);
	const float v = (static_cast<float>(sy) + r1) * (1.0f / scrsize.y);
	const glm::vec3 pointOnPixel =
		glm::vec3(p1.x, p1.y, p1.z) + u * glm::vec3(right.x, right.y, right.z) + v * glm::vec3(up.x, up.y, up.z);
	const glm::vec3 direction = normalize(pointOnPixel - origin);

	const uint bufferIdx = pathIdx + (bufferIndex * stride);

	pathOrigins[bufferIdx] = glm::vec4(origin, __uint_as_float((pathIdx << 8) + 1 /* 1 == specular */));
	pathDirections[bufferIdx] = glm::vec4(direction, 0);

	glm::vec4 result = glm::vec4(0, 0, __int_as_float(-1), 0);

	const float3 O = make_float3(origin.x, origin.y, origin.z);
	const float3 D = make_float3(direction.x, direction.y, direction.z);

	rtTrace(sceneRoot, make_Ray(O, D, 0u, 10.0f * geometryEpsilon, RT_DEFAULT_MAX), result);
	pathStates[bufferIdx] = result;
}

RT_PROGRAM void generateSecondaryRay()
{
	const uint pathIdx = launch_index % (scrsize.x * scrsize.y);
	const uint bufferIndex = (pathLength % 2);
	const uint bufferIdx = pathIdx + (bufferIndex * stride);

	const glm::vec4 O4 = pathOrigins[bufferIdx];
	const glm::vec4 D4 = pathDirections[bufferIdx];

	const float3 O = make_float3(O4.x, O4.y, O4.z);
	const float3 D = make_float3(D4.x, D4.y, D4.z);

	glm::vec4 result = glm::vec4(0, 0, __int_as_float(-1), 0);
	rtTrace(sceneRoot, make_Ray(O, D, 1u, 10.0f * geometryEpsilon, RT_DEFAULT_MAX), result);
	pathStates[bufferIdx] = result;
}

RT_PROGRAM void generateShadowRay()
{
	const uint pathIdx = launch_index % (scrsize.x * scrsize.y);
	const glm::vec4 O4 = connectData[pathIdx].Origin;
	const glm::vec4 D4 = connectData[pathIdx].Direction;

	const float3 O = make_float3(O4.x, O4.y, O4.z);
	const float3 D = make_float3(D4.x, D4.y, D4.z);

	uint isVisible = 0;
	const auto epsilon = 10.0f * geometryEpsilon;
	rtTrace(sceneRoot, make_Ray(O, D, 2u, epsilon, D4.w - 2.0f * epsilon), isVisible);
	if (isVisible == 0)
		return;

	const glm::vec4 contribution = connectData[launch_index].Emission;
	const uint pixelIdx = __float_as_uint(contribution.w);
	accumulator[pixelIdx] += glm::vec4(contribution.x, contribution.y, contribution.z, 1.0f);
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

	hit_data = glm::vec4(__uint_as_float(barycentrics), __uint_as_float(instanceIdx), __int_as_float(primIdx), t_hit);
}