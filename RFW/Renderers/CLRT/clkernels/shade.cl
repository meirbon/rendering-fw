#include "shared.h"

#define AREA_LIGHT_COUNT lightCount.x
#define POINT_LIGHT_COUNT lightCount.y
#define SPOT_LIGHT_COUNT lightCount.z
#define DIR_LIGHT_COUNT lightCount.w

kernel void shade_rays(global float4 *accumulator,							   // 0
					   uint pathLength,										   // 1
					   uint pathCount,										   // 2
					   uint stride,											   // 3
					   global uint *blueNoise,								   // 4
					   global CLPotentialContribution *potentialContributions, // 5
					   global float4 *states,								   // 6
					   global float4 *origins,								   // 7
					   global float4 *directions,							   // 8
					   global float4 *skybox,								   // 9
					   uint s_width,										   // 10
					   uint s_height,										   // 11
					   global CLMaterial *material,							   // 12
					   global uint *uintTextures,							   // 13
					   global float4 *floatTextures,						   // 14
					   global uint *counters,								   // 15
					   uint4 lightCount,									   // 16
					   global CLDeviceAreaLight *areaLights,				   // 17
					   global CLDevicePointLight *pointLights,				   // 18
					   global CLDeviceSpotLight *spotLights,				   // 19
					   global CLDeviceDirectionalLight *dirLights			   // 20
)
{
	const uint pathIdx = (uint)get_global_id(0);
	if (pathIdx >= pathCount)
		return; // Check bounds

	const uint shadeBufferOffset = (pathLength % 2) * stride; // Buffers used to shade current rays
	const uint shadeBufferIndex = shadeBufferIndex + pathIdx; // Buffers used to shade current rays

	const float4 O = origins[shadeBufferIndex];
	const float4 D = directions[shadeBufferIndex];

	const uint pixelID = (as_uint(O.w) >> 8u);

	accumulator[pathIdx] = (float4)(D.xyz, 1.0f);
}