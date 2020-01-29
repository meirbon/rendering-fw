#pragma once

#include "compat.h"

// To implement a bsdf, implement the following 2 functions:
// INLINE_FUNC vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 B, const vec3
// wo, const vec3 wi, REFERENCE_OF(float) pdf, REFERENCE_OF(uint) seed)
//{
//	pdf = 0.0f;
//	return vec3(0.0f);
//}

// INLINE_FUNC vec3 SampleBSDF(const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T, const vec3 B,
// const vec3 wi, const float t,
//							const bool backfacing, REFERENCE_OF(vec3) wo, REFERENCE_OF(float) pdf, REFERENCE_OF(uint)
//seed)
//{
//	return vec3(0.0f);
//}

//#include "lambert.h"
#include "disney.h"
//#include "microfacet.h"
//#include "microsurface_scattering.h"