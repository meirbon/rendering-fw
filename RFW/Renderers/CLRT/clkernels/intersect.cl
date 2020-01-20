#include "shared.h"

kernel void intersect_rays(uint pathLength,										   // 0
						   uint pathCount,										   // 1
						   uint stride,											   // 2
						   global CLPotentialContribution *potentialContributions, // 3
						   global float4 *states,								   // 4
						   global float4 *origins,								   // 5
						   global float4 *directions,							   // 6
						   uint phase,											   // 7
						   global float4 *accumulator							   // 8
)
{
	const uint pathIdx = (uint)get_global_id(0);
	if (pathIdx >= pathCount)
		return; // Check bounds

	const uint shadeBufferOffset = (pathLength % 2) * stride; // Buffers used to shade current rays
	const uint shadeBufferIndex = shadeBufferIndex + pathIdx; // Buffers used to shade current rays

	if (phase == STAGE_PRIMARY_RAY || phase == STAGE_SECONDARY_RAY)
	{
		const float4 O4 = origins[shadeBufferIndex];
		const float4 D4 = directions[shadeBufferIndex];
		float4 result = (float4)(0, 0, as_float(-1), 0);

		//states[shadeBufferIndex] = result;
	}
	else if (phase == STAGE_SHADOW_RAY)
	{
		const float3 O = potentialContributions[pathIdx].Origin.xyz;
		const float4 D = potentialContributions[pathIdx].Direction;

		int occluded = 1;

		// TODO: Trace ray

		if (occluded == 1)
		{
			return;
		}

		const float4 E_Idx = potentialContributions[pathIdx].Emission_pixelIdx;
		const uint pixelIdx = as_uint(E_Idx.w);
		accumulator[pixelIdx] += (float4)(E_Idx.xyz, 0.0);
	}
}