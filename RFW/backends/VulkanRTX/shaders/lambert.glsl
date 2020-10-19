#ifndef LAMBERT_H
#define LAMBERT_H

#include "tools.glsl"
#include "structures.glsl"

vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 B, const vec3 wo, const vec3 wi,
				  inout float pdf)
{
	pdf = abs(dot(wi, iN)) * INVPI;
	return shadingData.color.xyz * INVPI;
}

vec3 SampleBSDF(const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T, const vec3 B, const vec3 wo,
				const float t, const bool backfacing, const float r0, const float r1, inout vec3 wi, inout float pdf, inout bool specular)
{
	if (abs(ROUGHNESS) < 0.1f)
	{
		specular = true;
		wi = reflect(-wo, iN);
		pdf = 1.0f;
		if (dot(N, wi) <= 0.0f)
			pdf = 0.0f;
		return shadingData.color.xyz * (1.0f / (abs(dot(iN, wi))));
	}

	specular = false;
	wi = normalize(tangentToWorld(DiffuseReflectionUniform(r0, r1), iN, T, B));
	pdf = max(0.0f, dot(wi, iN)) * INVPI;
	if (dot(N, wi) <= 0)
		pdf = 0;
	return shadingData.color.xyz * INVPI;
}

#endif
