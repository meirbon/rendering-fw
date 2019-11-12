#pragma once

#include <glm/glm.hpp>
#include "tools.h"

// for debugging: Lambert brdf
__device__ static vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 wo,
									const vec3 wi, float &pdf)
{
	pdf = abs(dot(wi, iN)) * glm::one_over_pi<float>();
	return shadingData.color * glm::one_over_pi<float>();
}

__device__ static vec3 SampleBSDF(const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T,
								  const vec3 B, const vec3 wo, const float r3, const float r4, vec3 &wi, float &pdf)
{
	// specular and diffuse
	const float roughness = shadingData.getRoughness();
	if (roughness < 0.1f)
	{
		// pure specular
		wi = reflect(-wo, iN);

		if (dot(N, wi) <= 0.0f)
		{
			pdf = 0.0f;
			return vec3(0.0f);
		}

		pdf = 1.0f;
		return shadingData.color * (1.0f / abs(dot(iN, wi)));
	}

	wi = normalize(tangent2World(DiffuseReflectionCosWeighted(r3, r4), T, B, iN));
	if (dot(N, wi) <= 0.0f)
	{
		pdf = 0.0f;
		return vec3(0.0f);
	}

	pdf = max(0.0f, dot(wi, iN)) * glm::one_over_pi<float>();
	return shadingData.color * glm::one_over_pi<float>();
}

// EOF