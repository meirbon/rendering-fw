#pragma once

#include <DeviceStructures.h>
#include <glm/glm.hpp>
using namespace glm;
using namespace rfw;

__constant__ __device__ rfw::DeviceAreaLight *areaLights;
__constant__ __device__ rfw::DevicePointLight *pointLights;
__constant__ __device__ rfw::DeviceSpotLight *spotLights;
__constant__ __device__ rfw::DeviceDirectionalLight *directionalLights;
__constant__ __device__ LightCount lightCounts;

__device__ float PotentialAreaLightContribution(const int idx, const vec3 O, const vec3 N, const vec3 I, const vec3 bary)
{
	const DeviceAreaLight light = areaLights[idx];
	const vec4 posEnergy = light.pos_energy;
	const vec3 LN = light.getNormal();
	vec3 L = I;
	if (bary.x >= 0)
	{
		const vec3 V0 = light.getVertex0(); // vertex0
		const vec3 V1 = light.getVertex1(); // vertex1
		const vec3 V2 = light.getVertex2(); // vertex2
		L = bary.x * V0 + bary.y * V1 + bary.z * V2;
	}
	L = L - O;
	const float att = 1.0f / dot(L, L);
	L = normalize(L);
	const float LNdotL = max(0.0f, -dot(LN, L));
	const float NdotL = max(0.0f, dot(N, L));
	return posEnergy.w * LNdotL * NdotL * att;
}

__device__ float PotentialPointLightContribution(const int idx, const vec3 I, const vec3 N)
{
	const DevicePointLight light = pointLights[idx];
	const vec3 position = light.getPosition();
	const vec3 L = position - I;
	const float NdotL = max(0.0f, dot(N, L));
	const float att = 1.0f / dot(L, L);
	return light.getEnergy() * NdotL * att;
}

__device__ float PotentialSpotLightContribution(const int idx, const vec3 I, const vec3 N)
{
	const DeviceSpotLight light = spotLights[idx];
	const vec3 position = light.getPosition();
	const vec3 radiance = light.getRadiance();
	const vec3 direction = light.getDirection();
	const float cosOuter = light.getCosOuter();
	const float cosInner = light.getCosInner();

	vec3 L = position - I;
	const float att = 1.0f / dot(L, L);
	L = normalize(L);
	const float d = (max(0.0f, -dot(L, direction)) - cosOuter) / (cosInner - cosOuter);
	const float NdotL = max(0.0f, dot(N, L));
	const float LNdotL = max(0.0f, min(1.0f, d));
	return light.getEnergy() * LNdotL * NdotL * att;
	// TODO: other lights have radiance4.x+y+z precalculated as 'float energy'. For spots, this
	// does not help, as we need position4.w and direction4.w for the inner and outer angle anyway,
	// so we are touching 4 float4's. If we reduce the inner and outer angles to 16-bit values
	// however, the precalculated energy helps once more, and one float4 read disappears.
}

__device__ float PotentialDirectionalLightContribution(const int idx, const vec3 I, const vec3 N)
{
	const DeviceDirectionalLight light = directionalLights[idx];
	const vec3 direction = light.getDirection();
	const float LNdotL = max(0.0f, -dot(direction, N));
	return light.getEnergy() * LNdotL;
}

__device__ float CalculateLightPDF(const vec3 D, const float t, const float lightArea, const vec3 lightNormal)
{
	return (t * t) / (-dot(D, lightNormal) * lightArea);
}

__device__ float LightPickProb(const int idx, const vec3 O, const vec3 N, const vec3 I)
{
#if IS_LIGHTS
	// for implicit connections; calculates the chance that the light would have been explicitly selected
	float potential[MAX_IS_LIGHTS];
	float sum = 0;
	for (int i = 0; i < lightCounts.areaLightCount; i++)
	{
		float c = PotentialAreaLightContribution(i, O, N, I, vec3(-1));
		potential[i] = c;
		sum += c;
	}
	for (int i = 0; i < lightCounts.pointLightCount; i++)
	{
		float c = PotentialPointLightContribution(i, O, N);
		sum += c;
	}
	for (int i = 0; i < lightCounts.spotLightCount; i++)
	{
		float c = PotentialSpotLightContribution(i, O, N);
		sum += c;
	}
	for (int i = 0; i < lightCounts.directionalLightCount; i++)
	{
		float c = PotentialDirectionalLightContribution(i, O, N);
		sum += c;
	}
	if (sum <= 0)
		return 0; // no potential lights found
	return potential[idx] / sum;
#else
	return 1.0f / (lightCounts.areaLightCount + lightCounts.pointLightCount + lightCounts.spotLightCount + lightCounts.directionalLightCount);
#endif
}

// https://pharr.org/matt/blog/2019/02/27/triangle-sampling-1.html
__device__ vec3 RandomBarycentrics(const float r0)
{
	const uint uf = uint(r0 * uint(4294967295));
	vec2 A = vec2(1.f, 0.f);
	vec2 B = vec2(0.f, 1.f);
	vec2 C = vec2(0.f, 0.f);

	for (int i = 0; i < 16; ++i)
	{
		const int d = int((uf >> (2 * (15 - i))) & 0x3);
		vec2 An, Bn, Cn;
		switch (d)
		{
		case 0:
			An = (B + C) * 0.5f;
			Bn = (A + C) * 0.5f;
			Cn = (A + B) * 0.5f;
			break;
		case 1:
			An = A;
			Bn = (A + B) * 0.5f;
			Cn = (A + C) * 0.5f;
			break;
		case 2:
			An = (B + A) * 0.5f;
			Bn = B;
			Cn = (B + C) * 0.5f;
			break;
		case 3:
			An = (C + A) * 0.5f;
			Bn = (C + B) * 0.5f;
			Cn = C;
			break;
		}
		A = An, B = Bn, C = Cn;
	}
	const vec2 r = (A + B + C) * 0.3333333f;
	return vec3(r.x, r.y, 1.0f - r.x - r.y);
}

__device__ vec3 RandomPointOnLight(float r0, float r1, const vec3 I, const vec3 N, float &pickProb, float &lightPdf, vec3 &lightColor)
{
	const float lightCount = float(lightCounts.areaLightCount + lightCounts.pointLightCount + lightCounts.spotLightCount + lightCounts.directionalLightCount);
	const vec3 bary = RandomBarycentrics(r0);
#if IS_LIGHTS
	// importance sampling of lights, pickProb is per-light probability
	float potential[MAX_IS_LIGHTS];
	float sum = 0, total = 0;
	int lights = 0, lightIdx = 0;
	for (uint i = 0; i < lightCounts.areaLightCount; i++)
	{
		float c = PotentialAreaLightContribution(i, I, N, vec3(0), bary);
		potential[lights++] = c;
		sum += c;
	}
	for (uint i = 0; i < lightCounts.pointLightCount; i++)
	{
		float c = PotentialPointLightContribution(i, I, N);
		potential[lights++] = c;
		sum += c;
	}
	for (uint i = 0; i < lightCounts.spotLightCount; i++)
	{
		float c = PotentialSpotLightContribution(i, I, N);
		potential[lights++] = c;
		sum += c;
	}
	for (uint i = 0; i < lightCounts.directionalLightCount; i++)
	{
		float c = PotentialDirectionalLightContribution(i, I, N);
		potential[lights++] = c;
		sum += c;
	}
	if (sum <= 0) // no potential lights found
	{
		lightPdf = 0;
		return vec3(1.0f /* light direction; don't return 0 or nan, this will be slow */);
	}
	r1 *= sum;
	for (int i = 0; i < lights; i++)
	{
		total += potential[i];
		if (total >= r1)
		{
			lightIdx = i;
			break;
		}
	}
	pickProb = potential[lightIdx] / sum;
#else
	// uniform random sampling of lights, pickProb is simply 1.0 / lightCount
	pickProb = 1.0f / lightCount;
	int lightIdx = int(r0 * lightCount);
	r0 = (r0 - float(lightIdx) * (1.0f / lightCount)) * lightCount;
#endif
	lightIdx = clamp(lightIdx, 0, int(lightCount) - 1);
	if (lightIdx < lightCounts.areaLightCount)
	{
		const DeviceAreaLight light = areaLights[lightIdx];
		const vec3 V0 = light.getVertex0();
		const vec3 V1 = light.getVertex1();
		const vec3 V2 = light.getVertex2();
		lightColor = light.getRadiance();  // radiance
		const vec3 LN = light.getNormal(); // N
		const vec3 P = bary.x * V0 + bary.y * V1 + bary.z * V2;
		vec3 L = I - P; // reversed: from light to intersection point
		const float sqDist = dot(L, L);
		L = normalize(L);
		const float LNdotL = dot(L, LN);
		const float reciSolidAngle = sqDist / (light.getArea() * LNdotL); // LN.w contains area
		lightPdf = (LNdotL > 0 && dot(L, N) < 0) ? (reciSolidAngle * (1.0f / light.getEnergy())) : 0;
		return P;
	}

	if (lightIdx < (lightCounts.areaLightCount + lightCounts.pointLightCount))
	{
		const DevicePointLight light = pointLights[lightIdx - lightCounts.areaLightCount];
		const vec3 pos = light.getPosition(); // position
		lightColor = light.getRadiance();	  // radiance
		const vec3 L = I - pos;				  // reversed
		const float sqDist = dot(L, L);
		lightPdf = dot(L, N) < 0 ? (sqDist / light.getEnergy()) : 0;
		return pos;
	}

	if (lightIdx < (lightCounts.areaLightCount + lightCounts.pointLightCount + lightCounts.spotLightCount))
	{
		const DeviceSpotLight light = spotLights[lightIdx - (lightCounts.areaLightCount + lightCounts.pointLightCount)];
		const vec3 P = light.getPosition();
		const vec3 D = light.getDirection();
		vec3 L = I - P;
		const float sqDist = dot(L, L);
		L = normalize(L);
		const float d = max(0.0f, dot(L, D) - light.getCosOuter()) / (light.getCosInner() - light.getCosOuter());
		const float LNdotL = min(1.0f, d);
		lightPdf = (LNdotL > 0 && dot(L, N) < 0) ? (sqDist / (LNdotL * light.getEnergy())) : 0;
		lightColor = light.getRadiance();
		return P;
	}

	const DeviceDirectionalLight light = directionalLights[lightIdx - (lightCounts.areaLightCount + lightCounts.pointLightCount + lightCounts.spotLightCount)];
	const vec3 L = light.getDirection();
	lightColor = light.getRadiance();
	const float NdotL = dot(L, N);
	lightPdf = NdotL < 0 ? (1 * (1.0 / light.getEnergy())) : 0;
	return I - 1000.0f * L;
}
