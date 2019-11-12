#ifndef COUNTERS_H
#define COUNTERS_H

#include "../bindings.h"

layout( set = 0, binding = cCOUNTERS ) buffer Counters
{
	uint pathLength;
	uint scrWidth;
	uint scrHeight;
	uint bufferSize;
	uint activePaths;
	uint shaded;
	uint generated;
	uint connected;
	uint extended;
	uint extensionRays;
	uint shadowRays;
	uint totalExtensionRays;
	uint totalShadowRays;
	int probedInstid;
	int probedTriid;
	float probedDist;
	float clampValue;
	float geometryEpsilon;
	uvec4 lightCounts;
};

#endif