#pragma once

#include "../../RenderContext/DeviceStructures.h"

#define ALLOW_DENOISER 1

struct Counters
{
	unsigned int activePaths;
	unsigned int shaded;
	unsigned int extensionRays;
	unsigned int shadowRays;

	unsigned int totalExtensionRays;
	unsigned int totalShadowRays;
	unsigned int samplesTaken;
	unsigned int probeIdx;

	int probedInstanceId;
	int probedPrimId;
	float probedDistance;
	float dummy0;

	glm::vec3 probedPoint;
	float dummy1;
};

struct PotentialContribution
{
	glm::vec4 Origin;
	glm::vec4 Direction;
	glm::vec4 Emission;
};

__host__ __device__ static inline void createTangentSpace(const glm::vec3 &N, glm::vec3 &T, glm::vec3 &B)
{
	const float sign = copysignf(1.0f, N.z);
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	T = glm::vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
	B = glm::vec3(b, sign + N.y * N.y * a, -N.y);
}

__host__ __device__ static inline glm::vec3 tangentToWorld(const glm::vec3 &sample, const glm::vec3 &N, const glm::vec3 &T, const glm::vec3 &B)
{
	return glm::vec3(dot(T, sample), dot(B, sample), dot(N, sample));
}

__host__ __device__ static inline glm::vec3 worldToTangent(const glm::vec3 &sample, const glm::vec3 &N, const glm::vec3 &T, const glm::vec3 &B)
{
	return T * sample.x + B * sample.y + N * sample.z;
}
