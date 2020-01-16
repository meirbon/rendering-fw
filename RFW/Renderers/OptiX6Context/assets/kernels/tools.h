#pragma once

#include <glm/glm.hpp>

using namespace glm;

// from: https://aras-p.info/texts/CompactNormalStorage.html
__host__ __device__ inline uint PackNormal(const vec3 &N)
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
__host__ __device__ inline vec3 UnpackNormal(const uint p)
{
	vec4 nn = vec4((float)(p & 65535) * (2.0f / 65535.0f), (float)(p >> 16) * (2.0f / 65535.0f), 0, 0);
	nn += vec4(-1, -1, 1, -1);
	float l = dot(vec3(nn.x, nn.y, nn.z), vec3(-nn.x, -nn.y, -nn.w));
	nn.z = l, l = sqrtf(l), nn.x *= l, nn.y *= l;
	return vec3(nn) * 2.0f + vec3(0, 0, -1);
}

// alternative method
__host__ __device__ inline uint PackNormal2(const vec3 &N)
{
	// simple, and good enough discrimination of normals for filtering.
	const uint x = clamp((uint)((N.x + 1) * 511), 0u, 1023u);
	const uint y = clamp((uint)((N.y + 1) * 511), 0u, 1023u);
	const uint z = clamp((uint)((N.z + 1) * 511), 0u, 1023u);
	return (x << 2u) + (y << 12u) + (z << 22u);
}
__host__ __device__ inline vec3 UnpackNormal2(uint pi)
{
	const uint x = (pi >> 2u) & 1023u;
	const uint y = (pi >> 12u) & 1023u;
	const uint z = pi >> 22u;
	return vec3(x * (1.0f / 511.0f) - 1, y * (1.0f / 511.0f) - 1, z * (1.0f / 511.0f) - 1);
}

// color conversions

__host__ __device__ inline vec3 RGBToYCoCg(const vec3 RGB)
{
	const vec3 rgb = min(vec3(4), RGB); // clamp helps AA for strong HDR
	const float Y = dot(rgb, vec3(1, 2, 1)) * 0.25f;
	const float Co = dot(rgb, vec3(2, 0, -2)) * 0.25f + (0.5f * 256.0f / 255.0f);
	const float Cg = dot(rgb, vec3(-1, 2, -1)) * 0.25f + (0.5f * 256.0f / 255.0f);
	return vec3(Y, Co, Cg);
}

__host__ __device__ inline vec3 YCoCgToRGB(const vec3 YCoCg)
{
	const float Y = YCoCg.x;
	const float Co = YCoCg.y - (0.5f * 256.0f / 255.0f);
	const float Cg = YCoCg.z - (0.5f * 256.0f / 255.0f);
	return vec3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

__host__ __device__ inline float Luminance(vec3 rgb) { return 0.299f * min(rgb.x, 10.0f) + 0.587f * min(rgb.y, 10.0f) + 0.114f * min(rgb.z, 10.0f); }

__host__ __device__ inline uint HDRtoRGB32(const vec3 &c)
{
	const uint r = (uint)(1023.0f * min(1.0f, c.x));
	const uint g = (uint)(2047.0f * min(1.0f, c.y));
	const uint b = (uint)(2047.0f * min(1.0f, c.z));
	return (r << 22) + (g << 11) + b;
}

__host__ __device__ inline vec3 RGB32toHDR(const uint c)
{
	return vec3((float)(c >> 22) * (1.0f / 1023.0f), (float)((c >> 11) & 2047) * (1.0f / 2047.0f), (float)(c & 2047) * (1.0f / 2047.0f));
}
__host__ __device__ inline vec3 RGB32toHDRmin1(const uint c)
{
	return vec3((float)max(1u, c >> 22) * (1.0f / 1023.0f), (float)max(1u, (c >> 11) & 2047) * (1.0f / 2047.0f), (float)max(1u, c & 2047) * (1.0f / 2047.0f));
}

__host__ __device__ inline float SurvivalProbability(const vec3 &diffuse) { return min(1.0f, max(max(diffuse.x, diffuse.y), diffuse.z)); }

__host__ __device__ inline float FresnelDielectricExact(const vec3 &wo, const vec3 &N, float eta)
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

__host__ __device__ inline vec3 Tangent2World(const vec3 &V, const vec3 &N)
{
	vec3 B, T;
#if 0
	// "Building an Orthonormal Basis, Revisited"
	float sign = copysignf(1.0f, N.z);
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	B = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
	T = vec3(b, sign + N.y * N.y * a, -N.y);
#else
	if (N.z < 0.)
	{
		const float a = 1.0f / (1.0f - N.z);
		const float b = N.x * N.y * a;
		B = vec3(1.0f - N.x * N.x * a, -b, N.x);
		T = vec3(b, N.y * N.y * a - 1.0f, -N.y);
	}
	else
	{
		const float a = 1.0f / (1.0f + N.z);
		const float b = -N.x * N.y * a;
		B = vec3(1.0f - N.x * N.x * a, b, -N.x);
		T = vec3(b, 1.0f - N.y * N.y * a, -N.y);
	}
#endif
	return V.x * T + V.y * B + V.z * N;
}

__host__ __device__ inline vec3 tangent2World(const vec3 &sample, const vec3 &T, const vec3 &B, const vec3 &N)
{
	return sample.x * T + sample.y * B + sample.z * N;
}

__host__ __device__ inline vec3 World2Tangent(const vec3 &__restrict__ V, const vec3 &__restrict__ N)
{
	float sign = copysignf(1.0f, N.z);
	const float a = -1.0f / (sign + N.z);
	const float b = N.x * N.y * a;
	const vec3 B = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
	const vec3 T = vec3(b, sign + N.y * N.y * a, -N.y);
	return vec3(dot(V, T), dot(V, B), dot(V, N));
}

__device__ inline vec3 DiffuseReflectionUniform(const float r0, const float r1)
{
	const float term1 = glm::two_pi<float>() * r0, term2 = sqrt(1 - r1 * r1);
	float s = sin(term1);
	float c = cos(term1);
	return vec3(c * term2, s * term2, r1);
}

__device__ inline vec3 DiffuseReflectionCosWeighted(const float r0, const float r1)
{
	const float term1 = glm::two_pi<float>() * r0;
	const float term2 = sqrt(1.0 - r1);
	const float s = sin(term1);
	const float c = cos(term1);
	return vec3(c * term2, s * term2, sqrt(r1));
}

// origin offset

__host__ __device__ inline vec3 SafeOrigin(const vec3 &O, const vec3 &R, const vec3 &N, const float geoEpsilon)
{
	// offset outgoing ray direction along R and / or N: along N when strongly parallel to the origin surface; mostly
	// along R otherwise
	const float parallel = 1 - fabs(dot(N, R));
	const float v = parallel * parallel;
#if 1
	// we can go slightly into the surface when iN != N; negate the offset along N in that case
	const float side = dot(N, R) < 0 ? -1 : 1;
#else
	// negating offset along N only makes sense once we backface cull
	const float side = 1.0f;
#endif
	return O + R * geoEpsilon * (1.0f - v) + N * side * geoEpsilon * v;
}

// consistent normal interpolation

// vec3 ConsistentNormal(const vec3 &D, const vec3 &iN, const float alpha)
//{
//	// part of the implementation of "Consistent Normal Interpolation", Reshetov et al., 2010
//	// calculates a safe normal given an incoming direction, phong normal and alpha
//#ifndef CONSISTENTNORMALS
//	return iN;
//#else
//#if 0
//	// Eq. 1, exact
//	const float q = (1 - sinf( alpha )) / (1 + sinf( alpha ));
//#else
//	// Eq. 1 approximation, as in Figure 6 (not the wrong one in Table 8)
//	const float t = PI - 2 * alpha, q = (t * t) / (PI * (PI + (2 * PI - 4) * alpha));
//#endif
//	const float b = dot(D, iN), g = 1 + q * (b - 1), rho = sqrtf(q * (1 + g) / (1 + b));
//	const float3 Rc = (g + rho * b) * iN - (rho * D);
//	return normalize(D + Rc);
//#endif
//}

__device__ inline vec4 CombineToFloat4(const vec3 &A, const vec3 &B)
{
	// convert two float4's to a single uint4, where each int stores two components of the input vectors.
	// assumptions:
	// - the input is possitive
	// - the input can be safely clamped to 31.999
	// with this in mind, the data is stored as 5:11 unsigned fixed point, which should be plenty.
	const uint Ar = (uint)(min(A.x, 31.999f) * 2048.0f), Ag = (uint)(min(A.y, 31.999f) * 2048.0f), Ab = (uint)(min(A.z, 31.999f) * 2048.0f);
	const uint Br = (uint)(min(B.x, 31.999f) * 2048.0f), Bg = (uint)(min(B.y, 31.999f) * 2048.0f), Bb = (uint)(min(B.z, 31.999f) * 2048.0f);
	return vec4(__uint_as_float((Ar << 16) + Ag), __uint_as_float(Ab), __uint_as_float((Br << 16) + Bg), __uint_as_float(Bb));
}

__device__ inline vec3 GetDirectFromFloat4(const vec3 &X)
{
	const uint v0 = __float_as_uint(X.x), v1 = __float_as_uint(X.y);
	return vec3((float)(v0 >> 16) * (1.0f / 2048.0f), (float)(v0 & 65535) * (1.0f / 2048.0f), (float)v1 * (1.0f / 2048.0f));
}

__device__ inline vec3 GetIndirectFromFloat4(const vec4 &X)
{
	const uint v2 = __float_as_uint(X.z), v3 = __float_as_uint(X.w);
	return vec3((float)(v2 >> 16) * (1.0f / 2048.0f), (float)(v2 & 65535) * (1.0f / 2048.0f), (float)v3 * (1.0f / 2048.0f));
}

__host__ __device__ inline float blueNoiseSampler(const uint *blueNoise, int x, int y, int sampleIdx, int sampleDimension)
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

__host__ __device__ static inline void clampIntensity(vec3 &value, const float clampValue)
{
	const float v = max(value.x, max(value.y, value.z));
	if (v > clampValue)
	{
		const float m = clampValue / v;
		value = value * m;
	}
}
__host__ __device__ static inline void clampIntensity(vec4 &value, const float clampValue)
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