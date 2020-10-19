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