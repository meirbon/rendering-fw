#ifndef COMPAT_H
#define COMPAT_H

#include "../Settings.h"

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
#define FLT_MIN 1 .175494351e-38f
#endif

#if defined(__CUDACC__) || defined(__NVCC__) || defined(_WIN32) || defined(__linux__) || defined(__APPLE__)

#ifndef REFERENCE_OF
#define REFERENCE_OF(x) x &
#endif

#include <glm/glm.hpp>
#include <glm/ext.hpp>

using uint = glm::uint;
using namespace glm;

#if defined(__CUDACC__) || defined(__NVCC__)

#ifndef __device__
#define __device__
#endif

#ifndef INLINE_FUNC
#define INLINE_FUNC __device__ static inline
#endif

#ifndef MEMBER_INLINE_FUNC
#define MEMBER_INLINE_FUNC __device__ inline
#endif

#else

#ifndef INLINE_FUNC
#define INLINE_FUNC static inline
#endif

#ifndef MEMBER_INLINE_FUNC
#define MEMBER_INLINE_FUNC inline
#endif

#endif

template <uint S> INLINE_FUNC float char2flt(const unsigned int value)
{
	constexpr float scale = (1.0f / 255.0f);
	return static_cast<float>((value >> S) & 255) * scale;
}

struct ShadingData
{
	vec3 color;
	unsigned int flags;

	vec3 absorption;
	unsigned int matID;

	uvec4 parameters;

	MEMBER_INLINE_FUNC float getMetallic() const { return char2flt<0>(parameters.x); }
	MEMBER_INLINE_FUNC float getSubsurface() const { return char2flt<8>(parameters.x); }
	MEMBER_INLINE_FUNC float getSpecular() const { return char2flt<16>(parameters.x); }
	MEMBER_INLINE_FUNC float getRoughness() const { return max(0.001f, char2flt<24>(parameters.x)); }
	MEMBER_INLINE_FUNC float getSpecularTint() const { return char2flt<0>(parameters.y); }
	MEMBER_INLINE_FUNC float getAnisotropic() const { return char2flt<8>(parameters.y); }
	MEMBER_INLINE_FUNC float getSheen() const { return char2flt<16>(parameters.y); }
	MEMBER_INLINE_FUNC float getSheenTint() const { return char2flt<24>(parameters.y); }
	MEMBER_INLINE_FUNC float getClearCoat() const { return char2flt<0>(parameters.z); }
	MEMBER_INLINE_FUNC float getClearCoatingGloss() const { return char2flt<8>(parameters.z); }
	MEMBER_INLINE_FUNC float getTransmission() const { return char2flt<16>(parameters.z); }
	MEMBER_INLINE_FUNC float getEta() const { return char2flt<24>(parameters.z); }
	MEMBER_INLINE_FUNC float getCustom1() const { return char2flt<0>(parameters.w); }
	MEMBER_INLINE_FUNC float getCustom2() const { return char2flt<8>(parameters.w); }
	MEMBER_INLINE_FUNC float getCustom3() const { return char2flt<16>(parameters.w); }
	MEMBER_INLINE_FUNC float getCustom4() const { return char2flt<24>(parameters.w); }
	MEMBER_INLINE_FUNC bool isEmissive() const { return any(glm::greaterThan(color, vec3(1.0f))); }
};

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

#include "tools.h"
