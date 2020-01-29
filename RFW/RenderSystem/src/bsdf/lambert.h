#pragma once

#include <glm/glm.hpp>
#include "tools.h"
#include "compat.h"

// for debugging: Lambert brdf
INLINE_FUNC glm::vec3 EvaluateBSDF(const ShadingData shadingData, const glm::vec3 iN, const glm::vec3 T,
								   const glm::vec3 B, const glm::vec3 wo, const glm::vec3 wi, REFERENCE_OF(float) pdf)
{
	pdf = abs(dot(wi, iN)) * glm::one_over_pi<float>();
	return shadingData.color * glm::one_over_pi<float>();
}

INLINE_FUNC glm::vec3 SampleBSDF(const ShadingData shadingData, const glm::vec3 iN, const glm::vec3 N,
								 const glm::vec3 T, const glm::vec3 B, const glm::vec3 wo, const float t,
								 const bool backfacing, REFERENCE_OF(glm::vec3) wi, REFERENCE_OF(float) pdf, float r0,
								 float r1, float r2, float r3)
{
	// specular and diffuse
	const float roughness = ROUGHNESS;
	if (roughness < MIN_ROUGHNESS)
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
	else if (roughness < 1.0f)
	{
		if (r0 > roughness)
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
		else
		{
			wi = normalize(tangentToWorld(DiffuseReflectionCosWeighted(r1, r2), T, B, iN));
			if (dot(N, wi) <= 0.0f)
			{
				pdf = 0.0f;
				return vec3(0.0f);
			}

			pdf = max(0.0f, dot(wi, iN)) * glm::one_over_pi<float>();
			return shadingData.color * glm::one_over_pi<float>();
		}
	}
	else
	{
		wi = normalize(tangentToWorld(DiffuseReflectionCosWeighted(r0, r1), T, B, iN));
		if (dot(N, wi) <= 0.0f)
		{
			pdf = 0.0f;
			return vec3(0.0f);
		}

		pdf = max(0.0f, dot(wi, iN)) * glm::one_over_pi<float>();
		return shadingData.color * glm::one_over_pi<float>();
	}
}

// EOF