#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "tools.h"

using namespace glm;

// BeckmannDistribution Public Methods
__device__ __host__ inline float RoughnessToAlpha(float roughness)
{
	roughness = max(roughness, 1e-3f);
	const float x = log(roughness);
	return min(1.0f,
			   (1.62142f + 0.819955f * x + 0.1734f * x * x + 0.0171201f * x * x * x + 0.000640711f * x * x * x * x));
}

__device__ __host__ inline vec3 SphericalDirection(float sinTheta, float cosTheta, float phi)
{
	return vec3(sinTheta * cos(phi), sinTheta * sin(phi), cosTheta);
}

__device__ __host__ inline vec3 SphericalDirection(float sinTheta, float cosTheta, float phi, const vec3 &x,
												   const vec3 &y, const vec3 &z)
{
	return sinTheta * cosf(phi) * x + sinTheta * sinf(phi) * y + cosTheta * z;
}

__device__ __host__ inline float ErfInv(float x)
{
	float w, p;
	x = clamp(x, -.999f, .999f);
	w = -log((1.0f - x) * (1.0f + x));
	if (w < 5.0f)
	{
		w = w - 2.5f;
		p = 2.81022636e-08f;
		p = 3.43273939e-07f + p * w;
		p = -3.5233877e-06f + p * w;
		p = -4.39150654e-06f + p * w;
		p = 0.00021858087f + p * w;
		p = -0.00125372503f + p * w;
		p = -0.00417768164f + p * w;
		p = 0.246640727f + p * w;
		p = 1.50140941f + p * w;
	}
	else
	{
		w = sqrt(w) - 3.f;
		p = -0.000200214257f;
		p = 0.000100950558f + p * w;
		p = 0.00134934322f + p * w;
		p = -0.00367342844f + p * w;
		p = 0.00573950773f + p * w;
		p = -0.0076224613f + p * w;
		p = 0.00943887047f + p * w;
		p = 1.00167406f + p * w;
		p = 2.83297682f + p * w;
	}
	return p * x;
}

__device__ __host__ inline float Erf(float x)
{
	// Save the sign of x
	int sign = 1;
	if (x < 0.0f)
		sign = -1;
	x = fabs(x);

	// A&S formula 7.1.26
	const float t = 1.0f / (1.0f + 0.3275911f * x);
	const float y =
		1.0f - (((((1.061405429f * t + -1.453152027f) * t) + 1.421413741f) * t + -0.284496736f) * t + 0.254829592f) *
				   t * expf(-x * x);

	return sign * y;
}

__device__ __host__ inline float CosTheta(const vec3 &w) { return w.z; }

__device__ __host__ inline float Cos2Theta(const vec3 &w) { return w.z * w.z; }

__device__ __host__ inline float AbsCosTheta(const vec3 &w) { return fabs(w.z); }

__device__ __host__ inline float Sin2Theta(const vec3 &w) { return fmaxf(0.f, 1.f - Cos2Theta(w)); }

__device__ __host__ inline float SinTheta(const vec3 &w) { return sqrtf(Sin2Theta(w)); }

__device__ __host__ inline float TanTheta(const vec3 &w) { return SinTheta(w) / CosTheta(w); }

__device__ __host__ inline float Tan2Theta(const vec3 &w) { return Sin2Theta(w) / Cos2Theta(w); }

__device__ __host__ inline float CosPhi(const vec3 &w)
{
	float sinTheta = SinTheta(w);
	return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f);
}

__device__ __host__ inline float SinPhi(const vec3 &w)
{
	const float sinTheta = SinTheta(w);
	return (sinTheta == 0.0f) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f);
}

__device__ __host__ inline float Cos2Phi(const vec3 &w) { return CosPhi(w) * CosPhi(w); }

__device__ __host__ inline float Sin2Phi(const vec3 &w) { return SinPhi(w) * SinPhi(w); }

__device__ __host__ inline float CosDPhi(const vec3 &wa, const vec3 &wb)
{
	return clamp((wa.x * wb.x + wa.y * wb.y) / sqrtf((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)), -1.f,
				 1.f);
}

__device__ __host__ inline float G1(float lambda_w) { return 1.0f / (1.0f + lambda_w); }

__device__ __host__ inline float D(const vec3 &wh, float alphay, float alphax)
{
	const float tan2Theta = Tan2Theta(wh);
	if ((2.0f * tan2Theta) == tan2Theta)
		return 0.f;

	const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

	return expf(-tan2Theta * (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay))) /
		   (glm::pi<float>() * alphax * alphay * cos4Theta);
}

__device__ __host__ inline void BeckmannSample11(float cosThetaI, float r1, float r2, float *slope_x, float *slope_y)
{
	/* Special case (normal incidence) */
	if (cosThetaI > .9999f)
	{
		const float r = sqrtf(-logf(1.0f - r1));
		const float sinPhi = sinf(glm::two_pi<float>() * r2);
		const float cosPhi = cosf(glm::two_pi<float>() * r2);
		*slope_x = r * cosPhi;
		*slope_y = r * sinPhi;
		return;
	}

	/* The original inversion routine from the paper contained
	   discontinuities, which causes issues for QMC integration
	   and techniques like Kelemen-style MLT. The following code
	   performs a numerical inversion with better behavior */
	const float sinThetaI = sqrtf(fmaxf((float)0, (float)1 - cosThetaI * cosThetaI));
	const float tanThetaI = sinThetaI / cosThetaI;
	const float cotThetaI = 1 / tanThetaI;

	/* Search interval -- everything is parameterized
	   in the Erf() domain */
	float a = -1.0f;
	float c = Erf(cotThetaI);
	const float sample_x = fmaxf(r1, (float)1e-6f);

	/* Start with a good initial guess */
	// float b = (1-sample_x) * a + sample_x * c;

	/* We can do better (inverse of an approximation computed in
	 * Mathematica) */
	const float thetaI = acos(cosThetaI);
	const float fit = 1.0f + thetaI * (-0.876f + thetaI * (0.4265f - 0.0594f * thetaI));
	float b = c - (1.0f + c) * pow(1.0f - sample_x, fit);

	/* Normalization factor for the CDF */
	const float normalization =
		1.0f / (1.0f + c + 1.f / glm::root_two<float>() * tanThetaI * exp(-cotThetaI * cotThetaI));

	int it = 0;
	while (++it < 10)
	{
		/* Bisection criterion -- the oddly-looking
		   Boolean expression are intentional to check
		   for NaNs at little additional cost */
		if (!(b >= a && b <= c))
			b = 0.5f * (a + c);

		/* Evaluate the CDF and its derivative
		   (i.e. the density function) */
		const float invErf = ErfInv(b);
		const float value =
			normalization * (1.0f + b + 1.f / glm::root_two<float>() * tanThetaI * expf(-invErf * invErf)) - sample_x;
		float derivative = normalization * (1.f - invErf * tanThetaI);

		if (fabsf(value) < 1e-5f)
			break;

		/* Update bisection intervals */
		if (value > 0)
			c = b;
		else
			a = b;

		b -= value / derivative;
	}

	/* Now convert back into a slope value */
	*slope_x = ErfInv(b);

	/* Simulate Y component */
	*slope_y = ErfInv(2.0f * max(r2, (float)1e-6f) - 1.0f);
}

__device__ __host__ inline vec3 BeckmannSample(const vec3 &wi, float alpha_x, float alpha_y, float r1, float r2)
{
	// 1. stretch wi
	vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slope_x, slope_y;
	BeckmannSample11(CosTheta(wiStretched), r1, r2, &slope_x, &slope_y);

	// 3. rotate
	float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
	slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
	slope_x = tmp;

	// 4. unstretch
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	// 5. compute normal
	return normalize(vec3(-slope_x, -slope_y, 1.f));
}

__device__ __host__ inline void TrowbridgeReitzSample11(float cosTheta, float r1, float r2, float *slope_x,
														float *slope_y)
{
	// special case (normal incidence)
	if (cosTheta > .9999f)
	{
		float r = sqrtf(r1 / (1 - r1));
		float phi = 6.28318530718f * r2;
		*slope_x = r * cos(phi);
		*slope_y = r * sin(phi);
		return;
	}

	const float sinTheta = sqrtf(fmaxf(0.f, 1.f - cosTheta * cosTheta));
	const float tanTheta = sinTheta / cosTheta;
	float a = 1.f / tanTheta;
	const float G1 = 2.f / (1.f + sqrtf(1.f + 1.f / (a * a)));

	// sample slope_x
	const float A = 2.f * r1 / G1 - 1.f;
	float tmp = 1.f / (A * A - 1.f);
	if (tmp > 1e10)
		tmp = 1e10f;
	const float B = tanTheta;
	const float D = sqrtf(fmaxf(B * B * tmp * tmp - (A * A - B * B) * tmp, 0.0f));
	float slope_x_1 = B * tmp - D;
	float slope_x_2 = B * tmp + D;
	*slope_x = (A < 0.0f || slope_x_2 > 1.f / tanTheta) ? slope_x_1 : slope_x_2;

	// sample slope_y
	float S;
	if (r2 > 0.5f)
	{
		S = 1.f;
		r2 = 2.f * (r2 - .5f);
	}
	else
	{
		S = -1.f;
		r2 = 2.f * (.5f - r2);
	}
	float z = (r2 * (r2 * (r2 * 0.27385f - 0.73369f) + 0.46341f)) /
			  (r2 * (r2 * (r2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	*slope_y = S * z * sqrtf(1.f + *slope_x * *slope_x);
}

__device__ __host__ inline vec3 TrowbridgeReitzSample(const vec3 &wi, float alpha_x, float alpha_y, float r1, float r2)
{
	// 1. stretch wi
	const vec3 wiStretched = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slope_x, slope_y;
	TrowbridgeReitzSample11(CosTheta(wiStretched), r1, r2, &slope_x, &slope_y);

	// 3. rotate
	const float tmp = CosPhi(wiStretched) * slope_x - SinPhi(wiStretched) * slope_y;
	slope_y = SinPhi(wiStretched) * slope_x + CosPhi(wiStretched) * slope_y;
	slope_x = tmp;

	// 4. unstretch
	slope_x = alpha_x * slope_x;
	slope_y = alpha_y * slope_y;

	// 5. compute normal
	return normalize(vec3(-slope_x, -slope_y, 1.f));
}

__device__ __host__ inline void sampleGGX_P22_11(float cosThetaI, float *slopex, float *slopey, float r1, float r2)
{
	// The special case where the ray comes from normal direction
	// The following sampling is equivalent to the sampling of
	// micro facet normals (not slopes) on isotropic rough surface
	if (cosThetaI > 0.9999f)
	{
		const float r = sqrtf(r1 / (1.0f - r1));
		const float sinPhi = sinf(glm::two_over_pi<float>() * r2);
		const float cosPhi = cosf(glm::two_over_pi<float>() * r2);
		*slopex = r * cosPhi;
		*slopey = r * sinPhi;
		return;
	}

	const float sinThetaI = sqrt(max(0.0f, 1.0f - cosThetaI * cosThetaI));
	const float tanThetaI = sinThetaI / cosThetaI;
	const float a = 1.0f / tanThetaI;
	const float G1 = 2.0f / (1.0f + sqrt(1.0f + 1.0f / (a * a)));

	// Sample slope x
	const float A = 2.0f * r1 / G1 - 1.0f;
	const float B = tanThetaI;
	const float tmp = min(1.0f / (A * A - 1.0f), 1.0e12f);

	const float D = sqrt(B * B * tmp * tmp - (A * A - B * B) * tmp);
	const float slopex1 = B * tmp - D;
	const float slopex2 = B * tmp + D;
	*slopex = (A < 0.0f || slopex2 > 1.0f / tanThetaI) ? slopex1 : slopex2;

	// Sample slope y
	float S;
	if (r2 > 0.5f)
		S = 1.0f, r2 = 2.0f * (r2 - 0.5f);
	else
		S = -1.0f, r2 = 2.0f * (0.5f - r2);

	const float z = (r2 * (r2 * (r2 * 0.27385f - 0.73369f) + 0.46341f)) /
					(r2 * (r2 * (r2 * 0.093073f + 0.309420f) - 1.000000f) + 0.597999f);
	*slopey = S * z * sqrtf(1.0f + (*slopex) * (*slopex));
}

__device__ __host__ inline vec3 sampleGGX(const vec3 &wi, float alphaX, float alphaY, float r1, float r2)
{
	// E. Heitz, "Sampling the GGX Distribution of Visible Normals", 2018

	// 1. stretch wi
	const vec3 wiStretched = normalize(vec3(alphaX * wi.x, alphaY * wi.y, wi.z));

	// 2. simulate P22_{wi}(x_slope, y_slope, 1, 1)
	float slopex, slopey;
	sampleGGX_P22_11(CosTheta(wiStretched), &slopex, &slopey, r1, r2);

	// 3. rotate
	const float tmp = CosPhi(wiStretched) * slopex - SinPhi(wiStretched) * slopey;
	slopey = SinPhi(wiStretched) * slopex + CosPhi(wiStretched) * slopey;
	slopex = tmp;

	// 4. unstretch
	slopex = alphaX * slopex;
	slopey = alphaY * slopey;

	// 5. compute normal
	return normalize(vec3(-slopex, -slopey, 1.0f));
}

// GGX
__device__ __host__ inline vec3 sample_ggx(const vec3 &wo, float alphaX, float alphaY, float r1, float r2)
{
#if 0
			const bool flip = wo.z < 0.0f;
			const vec3 wm = sampleGGX(flip ? -wo : wo, alphaX, alphaY, r1, r2);

			if (wm.z < 0.0f)
				return -wm;

			return wm;
#else
	return sampleGGX(wo, alphaX, alphaY, r1, r2);
#endif
}

__device__ __host__ inline float lambda_ggx(const vec3 &wo, float alphaX, float alphaY)
{
	const float absTanThetaO = abs(TanTheta(wo));
	if (2.0f * absTanThetaO == absTanThetaO)
		return 0.0f;

	const float alpha = sqrt(Cos2Phi(wo) * alphaX * alphaX + Sin2Phi(wo) * alphaY * alphaY);
	const float alpha2Tan2Theta = alpha * absTanThetaO * alpha * absTanThetaO;
	return (-1.0f + sqrt(1.0f + alpha2Tan2Theta)) / 2.0f;
}

__device__ __host__ inline float pdf_ggx(const vec3 &wo, const vec3 &wh, const vec3 &wi, float alphaX, float alphaY)
{
	return G1(lambda_ggx(wo, alphaX, alphaY));
}

// BECKMANN
__device__ __host__ inline vec3 sample_beckmann(const vec3 &wo, float alphaX, float alphaY, float r1, float r2)
{
	// Sample visible area of normals for Beckmann distribution
	const bool flip = wo.z < 0.f;
	const vec3 wh = BeckmannSample(flip ? -wo : wo, alphaX, alphaY, r1, r2);

	if (flip)
		return -wh;
	return wh;
}

__device__ __host__ inline float lambda_beckmann(const vec3 &w, float alphaX, float alphaY)
{
	const float absTanTheta = abs(TanTheta(w));
	// Check for infinity
	if ((2.0f * absTanTheta) == absTanTheta)
		return 0.f;

	// Compute _alpha_ for direction _w_
	const float alpha = sqrt(Cos2Phi(w) * alphaX * alphaX + Sin2Phi(w) * alphaY * alphaY);
	const float a = 1.0f / (alpha * absTanTheta);

	if (a >= 1.6f)
		return 0.f;

	return (1.f - 1.259f * a + 0.396f * a * a) / (3.535f * a + 2.181f * a * a);
}

__device__ __host__ inline float pdf_beckmann(const vec3 &wo, const vec3 &wh, const vec3 &wi, float alphaX,
											  float alphaY)
{
	return G1(lambda_beckmann(wo, alphaX, alphaY));
}

__device__ __host__ inline vec3 sample_trowbridge_reitz(const vec3 &wo, float alphaX, float alphaY, float r1, float r2)
{
	float cosTheta = 0;
	float phi = glm::two_over_pi<float>() * r2;

#if 0
	//if (alphaX == alphaY)
	//{
		const float tanTheta2 = alphaX * alphaX * r1 / (1.0f - r1);
		cosTheta = 1.0f / sqrtf(1.0f + tanTheta2);
#else
	//}
	// else
	//{
	phi = atan(alphaY / alphaX * tan(glm::two_over_pi<float>() * r2 + .5f * glm::one_over_pi<float>()));
	if (r2 > .5f)
		phi += glm::one_over_pi<float>();
	const float sinPhi = sin(phi);
	const float cosPhi = cos(phi);
	const float alphax2 = alphaX * alphaX, alphay2 = alphaY * alphaY;
	const float alpha2 = 1.0f / (cosPhi * cosPhi / alphax2 + sinPhi * sinPhi / alphay2);
	const float tanTheta2 = alpha2 * r1 / (1.0f - r1);
	cosTheta = 1.0f / sqrt(1.0f + tanTheta2);
	//}
#endif

	const float sinTheta = sqrt(max(0.f, 1.f - cosTheta * cosTheta));
	const vec3 wh = SphericalDirection(sinTheta, cosTheta, phi);
	if (wh.z < 0.0f)
		return -wh;
	return wh;
}

__device__ __host__ inline float lambda_trowbridge_reitz(const vec3 &w, float alphaX, float alphaY)
{
	const float absTanTheta = abs(TanTheta(w));

	if ((2.0f * absTanTheta) == absTanTheta)
		return 0.f;

	// Compute _alpha_ for direction _w_
	const float alpha = sqrtf(Cos2Phi(w) * alphaX * alphaX + Sin2Phi(w) * alphaY * alphaY);
	const float alpha2Tan2Theta = (alpha * absTanTheta) * (alpha * absTanTheta);
	return (-1.f + sqrt(1.f + alpha2Tan2Theta)) / 2.f;
}

__device__ __host__ inline float pdf_trowbridge_reitz(const vec3 &wo, const vec3 &wh, const vec3 &wi, float alphaX,
													  float alphaY)
{
	return G1(lambda_trowbridge_reitz(wo, alphaX, alphaY));
}

__device__ static glm::vec3 EvaluateBSDF(const ShadingData &shadingData, const glm::vec3 &iN, const glm::vec3 &T,
										 const glm::vec3 &B, const glm::vec3 wo, const glm::vec3 wi, float &pdf)
{
	/*const glm::vec3 bsdf = BSDFEval(shadingData, iN, wo, wi);
	pdf = BSDFPdf(shadingData, iN, wo, wi);
	return bsdf;*/

	const float roughness = shadingData.getRoughness();
	if (roughness < 0.01f) // Use purely specular BRDF for roughness below threshold
	{
		pdf = 1.0f;
		return shadingData.color;
	}

	const vec3 wiLocal = worldToTangent(wi, iN, T, B);
	const vec3 iNLocal = worldToTangent(iN, iN, T, B);
	const vec3 woLocal = worldToTangent(wo, iN, T, B);

	pdf = 1.0f / pdf_ggx(wiLocal, iNLocal, woLocal, roughness, roughness);
	return shadingData.color;
}

__device__ static glm::vec3 SampleBSDF(const ShadingData &shadingData, const glm::vec3 &iN, const glm::vec3 &N,
									   const glm::vec3 &T, const glm::vec3 &B, const glm::vec3 &wo, const float r3,
									   const float r4, glm::vec3 &wi, float &pdf)
{
	const float roughness = shadingData.getRoughness();
	if (roughness < 0.01f) // Use purely specular BRDF for roughness below threshold
	{
		wi = reflect(-wo, iN);
		pdf = 1.0f;
		return shadingData.color;
	}

	const vec3 woLocal = worldToTangent(wo, iN, T, B);
	const vec3 sample = sampleGGX(woLocal, roughness, roughness, r3, r4);
	const vec3 wiLocal = reflect(-wo, sample);
	pdf = 1.0f / pdf_ggx(wiLocal, sample, woLocal, roughness, roughness);

	wi = tangent2World(wiLocal, T, B, iN);
	if (dot(wi, N) <= 0.0f)
		pdf = 0.0f;

	return shadingData.color;
}
