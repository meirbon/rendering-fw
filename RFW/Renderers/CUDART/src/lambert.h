#pragma once

#include <glm/glm.hpp>
#include "tools.h"
#include "compat.h"

// for debugging: Lambert brdf
INLINE_FUNC vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 wo, const vec3 wi, float &pdf)
{
	pdf = abs(dot(wi, iN)) * glm::one_over_pi<float>();
	return shadingData.color * glm::one_over_pi<float>();
}

INLINE_FUNC vec3 SampleBSDF(const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T, const vec3 B, const vec3 wo, vec3 &wi, float &pdf,
							REFERENCE_OF(uint) seed)
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

	wi = normalize(tangent2World(DiffuseReflectionCosWeighted(RandomFloat(seed), RandomFloat(seed)), T, B, iN));
	if (dot(N, wi) <= 0.0f)
	{
		pdf = 0.0f;
		return vec3(0.0f);
	}

	pdf = max(0.0f, dot(wi, iN)) * glm::one_over_pi<float>();
	return shadingData.color * glm::one_over_pi<float>();
}

// EOF