#pragma once

#include "compat.h"

// from: https://aras-p.info/texts/CompactNormalStorage.html
INLINE_FUNC uint PackNormal(const vec3 N)
{
#if 1
	// more efficient
	const float f = 65535.0f / max(sqrtf(8.0f * N.z + 8.0f), 0.0001f); // Thanks Robbin Marcus
	return (uint)(N.x * f + 32767.0f) + ((uint)(N.y * f + 32767.0f) << 16);
#else
	float2 enc = normalize(make_float2(N)) * (sqrtf(-N.z * 0.5f + 0.5f));
	enc = enc * 0.5f + 0.5f;
	return (uint)(enc.x * 65535.0f) + ((uint)(enc.y * 65535.0f) << 16);
#endif
}
INLINE_FUNC vec3 UnpackNormal(const uint p)
{
	vec4 nn = vec4((float)(p & 65535) * (2.0f / 65535.0f), (float)(p >> 16) * (2.0f / 65535.0f), 0, 0);
	nn += vec4(-1, -1, 1, -1);
	float l = dot(vec3(nn.x, nn.y, nn.z), vec3(-nn.x, -nn.y, -nn.w));
	nn.z = l, l = sqrtf(l), nn.x *= l, nn.y *= l;
	return vec3(nn) * 2.0f + vec3(0, 0, -1);
}

// alternative method
INLINE_FUNC uint PackNormal2(const vec3 N)
{
	// simple, and good enough discrimination of normals for filtering.
	const uint x = clamp((uint)((N.x + 1) * 511), 0u, 1023u);
	const uint y = clamp((uint)((N.y + 1) * 511), 0u, 1023u);
	const uint z = clamp((uint)((N.z + 1) * 511), 0u, 1023u);
	return (x << 2u) + (y << 12u) + (z << 22u);
}
INLINE_FUNC vec3 UnpackNormal2(uint pi)
{
	const uint x = (pi >> 2u) & 1023u;
	const uint y = (pi >> 12u) & 1023u;
	const uint z = pi >> 22u;
	return vec3(x * (1.0f / 511.0f) - 1, y * (1.0f / 511.0f) - 1, z * (1.0f / 511.0f) - 1);
}

// color conversions

INLINE_FUNC vec3 RGBToYCoCg(const vec3 RGB)
{
	const vec3 rgb = min(vec3(4), RGB); // clamp helps AA for strong HDR
	const float Y = dot(rgb, vec3(1, 2, 1)) * 0.25f;
	const float Co = dot(rgb, vec3(2, 0, -2)) * 0.25f + (0.5f * 256.0f / 255.0f);
	const float Cg = dot(rgb, vec3(-1, 2, -1)) * 0.25f + (0.5f * 256.0f / 255.0f);
	return vec3(Y, Co, Cg);
}

INLINE_FUNC vec3 YCoCgToRGB(const vec3 YCoCg)
{
	const float Y = YCoCg.x;
	const float Co = YCoCg.y - (0.5f * 256.0f / 255.0f);
	const float Cg = YCoCg.z - (0.5f * 256.0f / 255.0f);
	return vec3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

INLINE_FUNC float Luminance(vec3 rgb)
{
	return 0.299f * min(rgb.x, 10.0f) + 0.587f * min(rgb.y, 10.0f) + 0.114f * min(rgb.z, 10.0f);
}

INLINE_FUNC uint HDRtoRGB32(const vec3 &c)
{
	const uint r = (uint)(1023.0f * min(1.0f, c.x));
	const uint g = (uint)(2047.0f * min(1.0f, c.y));
	const uint b = (uint)(2047.0f * min(1.0f, c.z));
	return (r << 22) + (g << 11) + b;
}

INLINE_FUNC vec3 RGB32toHDR(const uint c)
{
	return vec3((float)(c >> 22) * (1.0f / 1023.0f), (float)((c >> 11) & 2047) * (1.0f / 2047.0f),
				(float)(c & 2047) * (1.0f / 2047.0f));
}
INLINE_FUNC vec3 RGB32toHDRmin1(const uint c)
{
	return vec3((float)max(1u, c >> 22) * (1.0f / 1023.0f), (float)max(1u, (c >> 11) & 2047) * (1.0f / 2047.0f),
				(float)max(1u, c & 2047) * (1.0f / 2047.0f));
}

INLINE_FUNC float SurvivalProbability(const vec3 diffuse)
{
	return min(1.0f, max(max(diffuse.x, diffuse.y), diffuse.z));
}

INLINE_FUNC float FresnelDielectricExact(const vec3 wo, const vec3 N, float eta)
{
	if (eta <= 1.0f)
		return 0.0f;
	const float cosThetaI = max(0.0f, dot(wo, N));
	float scale = 1.0f / eta, cosThetaTSqr = 1 - (1 - cosThetaI * cosThetaI) * (scale * scale);
	if (cosThetaTSqr <= 0.0f)
		return 1.0f;
	float cosThetaT = sqrtf(cosThetaTSqr);
	float Rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	float Rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (Rs * Rs + Rp * Rp);
}

INLINE_FUNC vec3 DiffuseReflectionUniform(const float r0, const float r1)
{
	const float term1 = glm::two_pi<float>() * r0, term2 = sqrt(1 - r1 * r1);
	float s = sin(term1);
	float c = cos(term1);
	return vec3(c * term2, s * term2, r1);
}

INLINE_FUNC vec3 DiffuseReflectionCosWeighted(const float r0, const float r1)
{
	const float term1 = glm::two_pi<float>() * r0;
	const float term2 = sqrt(1.0f - r1);
	const float s = sin(term1);
	const float c = cos(term1);
	return normalize(vec3(c * term2, s * term2, sqrt(r1)));
}

INLINE_FUNC vec3 SafeOrigin(const vec3 O, const vec3 R, const vec3 N, const float geoEpsilon)
{
#if 1
	return O + N * 1e-5f;
#else
	// offset outgoing ray direction along R and / or N: along N when strongly parallel to the origin surface; mostly
	// along R otherwise
	const float parallel = 1 - fabs(dot(N, R));
	const float v = parallel * parallel;
#if 1
	// we can go slightly into the surface when iN != N; negate the offset along N in that case
	const float side = dot(N, R) < 0.f ? -1.f : 1.f;
#else
	// negating offset along N only makes sense once we backface cull
	const float side = 1.0f;
#endif
	return O + R * geoEpsilon * (1.0f - v) + N * side * geoEpsilon * v;
#endif
}

INLINE_FUNC vec4 CombineToFloat4(const vec3 A, const vec3 B)
{
	// convert two float4's to a single uint4, where each int stores two components of the input vectors.
	// assumptions:
	// - the input is possitive
	// - the input can be safely clamped to 31.999
	// with this in mind, the data is stored as 5:11 unsigned fixed point, which should be plenty.
	const uint Ar = (uint)(min(A.x, 31.999f) * 2048.0f);
	const uint Ag = (uint)(min(A.y, 31.999f) * 2048.0f);
	const uint Ab = (uint)(min(A.z, 31.999f) * 2048.0f);
	const uint Br = (uint)(min(B.x, 31.999f) * 2048.0f);
	const uint Bg = (uint)(min(B.y, 31.999f) * 2048.0f);
	const uint Bb = (uint)(min(B.z, 31.999f) * 2048.0f);

	const uint x = (Ar << 16) + Ag;
	const uint y = Ab;
	const uint z = (Br << 16) + Bg;
	const uint w = Bb;
	return vec4(UINT_AS_FLOAT(x), UINT_AS_FLOAT(y), UINT_AS_FLOAT(z), UINT_AS_FLOAT(w));
}

INLINE_FUNC vec3 GetDirectFromFloat4(const vec3 X)
{
	const uint v0 = FLOAT_AS_UINT(X.x);
	const uint v1 = FLOAT_AS_UINT(X.y);
	return vec3((float)(v0 >> 16) * (1.0f / 2048.0f), (float)(v0 & 65535) * (1.0f / 2048.0f),
				(float)v1 * (1.0f / 2048.0f));
}

INLINE_FUNC vec3 GetIndirectFromFloat4(const vec4 X)
{
	const uint v2 = FLOAT_AS_UINT(X.z);
	const uint v3 = FLOAT_AS_UINT(X.w);
	return vec3((float)(v2 >> 16) * (1.0f / 2048.0f), (float)(v2 & 65535) * (1.0f / 2048.0f),
				(float)v3 * (1.0f / 2048.0f));
}

INLINE_FUNC float blueNoiseSampler(const uint *blueNoise, int x, int y, int sampleIdx, int sampleDimension)
{
	// wrap arguments
	x &= 127;
	y &= 127;
	sampleIdx &= 255;
	sampleDimension &= 255;

	// xor index based on optimized ranking
	const int rankedSampleIndex = sampleIdx ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3];

	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];

	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];

	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

INLINE_FUNC void clampIntensity(REFERENCE_OF(vec3) value, const float clampValue)
{
	const float v = max(value.x, max(value.y, value.z));
	if (v > clampValue)
	{
		const float m = clampValue / v;
		value = value * m;
	}
}
INLINE_FUNC void clampIntensity(REFERENCE_OF(vec4) value, const float clampValue)
{
	const float v = max(value.x, max(value.y, value.z));
	if (v > clampValue)
	{
		const float m = clampValue / v;
		value.x = value.x * m;
		value.y = value.y * m;
		value.z = value.z * m;
	}
}

INLINE_FUNC void createTangentSpace(const vec3 N, REFERENCE_OF(vec3) T, REFERENCE_OF(vec3) B)
{
#if 1 // Frisvad
	if (abs(N.x) > abs(N.z))
	{
		T = vec3(-N.y, N.x, 0.0f);
		B = vec3(0.0f, -N.z, N.y);
	}
	else
	{
		B = vec3(0.0f, -N.z, N.y);
		T = cross(B, N);
	}
#else
	const float s = sign(N.z);
	const float a = -1.0f / (s + N.z);
	const float b = N.x * N.y * a;
	T = vec3(1.0f + s * N.x * N.x * a, s * b, -s * N.x);
	B = vec3(b, s + N.y * N.y * a, -N.y);
#endif
}

INLINE_FUNC vec3 tangentToWorld(const vec3 s, const vec3 N, const vec3 T, const vec3 B)
{
	return T * s.x + B * s.y + N * s.z;
}

INLINE_FUNC vec3 worldToTangent(const vec3 s, const vec3 N, const vec3 T, const vec3 B)
{
	return vec3(dot(T, s), dot(B, s), dot(N, s));
}

INLINE_FUNC unsigned int WangHash(uint s)
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d, s = s ^ (s >> 15);
	return s;
}

INLINE_FUNC unsigned int RandomInt(REFERENCE_OF(uint) s)
{
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

INLINE_FUNC float RandomFloat(REFERENCE_OF(uint) s) { return RandomInt(s) * 2.3283064365387e-10f; }