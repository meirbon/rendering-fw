#ifndef COMPAT_H
#define COMPAT_H

// TODO: Move ShadingData struct to bsdf file (this file)
#include "getShadingData.h"
#include "random.h"

#define INVPI 0.318309886183790671537767526745028724f
#define PI 3.14159265358979323846264338327950288f
#define INV2PI 0.159154943091895335768883763372514362f
#define TWOPI 6.28318530717958647692528676655900576f
#define HALF_PI 0.31830988618379067153f
#define SQRT_PI 1.77245385090551602729f
#define INV_SQRT_PI 0.56418958354775628694f
#define HALF_INV_SQRT_PI 0.28209479177387814347f
#define INV_SQRT_2PI 0.3989422804014326779f
#define INV_SQRT_2 0.7071067811865475244f
#define SQRT_2 1.41421356237309504880f
#ifndef FLT_MAX
#define FLT_MAX 3.402823466e+38f
#endif
#ifndef FLT_MIN
#define 1.175494351e-38f
#endif

#if defined(__CUDACC__) || defined(_WIN32) || defined(__linux__)
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace glm;

#define INLINE_FUNC __device__ static inline
#define REFERENCE_OF(x) x &

#define ETA shadingData.getEta()
#define SPECULAR shadingData.getSpecular()
#define METALLIC shadingData.getMetallic()
#define TRANSMISSION shadingData.getTransmission()
#define ROUGHNESS shadingData.getRoughness()
#define SUBSURFACE shadingData.getSubsurface()
#define SPECTINT shadingData.getSpecularTint()
#define CLEARCOAT shadingData.getClearCoat()
#define CLEARCOATGLOSS shadingData.getClearCoatingGloss()
#else
#define INLINE_FUNC
#define REFERENCE_OF(x) inout x
#endif

#endif