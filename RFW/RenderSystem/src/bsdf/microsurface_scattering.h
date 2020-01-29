#pragma once

#include "compat.h"

namespace mf_slope
{
INLINE_FUNC float alpha_i(float alpha_x, float alpha_y, const vec3 wi);
INLINE_FUNC float D(const float alpha_x, const float alpha_y, const vec3 wm);
INLINE_FUNC float D_wi(const float alpha_x, const float alpha_y, const vec3 wi, const vec3 wm);
INLINE_FUNC vec3 sampleD_wi(const float alpha_x, const float alpha_y, const vec3 wi, const float U1, const float U2);
} // namespace mf_slope

// build orthonormal basis (Building an Orthonormal Basis from a 3D Unit Vector Without Normalization, [Frisvad2012])
INLINE_FUNC void buildOrthonormalBasis(REFERENCE_OF(vec3) omega_1, REFERENCE_OF(vec3) omega_2, const vec3 omega_3)
{
	if (omega_3.z < -0.9999999f)
	{
		omega_1 = vec3(0.0f, -1.0f, 0.0f);
		omega_2 = vec3(-1.0f, 0.0f, 0.0f);
	}
	else
	{
		const float a = 1.0f / (1.0f + omega_3.z);
		const float b = -omega_3.x * omega_3.y * a;
		omega_1 = vec3(1.0f - omega_3.x * omega_3.x * a, b, -omega_3.x);
		omega_2 = vec3(b, 1.0f - omega_3.y * omega_3.y * a, -omega_3.y);
	}
}

INLINE_FUNC bool is_finite_number(const float x) { return (x <= FLT_MAX & x >= -FLT_MAX); }

INLINE_FUNC float mf_erf(float x)
{
	// constants
	const float a1 = 0.254829592;
	const float a2 = -0.284496736;
	const float a3 = 1.421413741;
	const float a4 = -1.453152027;
	const float a5 = 1.061405429;
	const float p = 0.3275911;

	// Save the sign of x
	int sign = 1;
	if (x < 0)
		sign = -1;
	x = abs(x);

	// A&S formula 7.1.26
	const float t = 1.0f / (1.0f + p * x);
	const float y = 1.0f - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x);

	return sign * y;
}

INLINE_FUNC float mf_erfinv(float x)
{
	float w, p;
	w = -log((1.0f - x) * (1.0f + x));
	if (w < 5.000000f)
	{
		w = w - 2.500000f;
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
		w = sqrt(w) - 3.000000f;
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

INLINE_FUNC float abgam(float x)
{
	float gam[10], temp;

	gam[0] = 1.f / 12.f;
	gam[1] = 1.f / 30.f;
	gam[2] = 53.f / 210.f;
	gam[3] = 195.f / 371.f;
	gam[4] = 22999.f / 22737.f;
	gam[5] = 29944523.f / 19733142.f;
	gam[6] = 109535241009.f / 48264275462.f;
	temp = 0.5 * log(2 * PI) - x + (x - 0.5f) * log(x) +
		   gam[0] / (x + gam[1] / (x + gam[2] / (x + gam[3] / (x + gam[4] / (x + gam[5] / (x + gam[6] / x))))));

	return temp;
}

INLINE_FUNC float gamma(float x)
{
	float result;
	result = exp(abgam(x + 5)) / (x * (x + 1) * (x + 2) * (x + 3) * (x + 4));
	return result;
}

INLINE_FUNC float beta(float m, float n) { return (gamma(m) * gamma(n) / gamma(m + n)); }

namespace mf_height_uniform
{
// height PDF
INLINE_FUNC float P1(const float h) { return (h >= -1.0f && h <= 1.0f) ? 0.5f : 0.0f; }

// height CDF
INLINE_FUNC float C1(const float h) { return min(1.0f, max(0.0f, 0.5f * (h + 1.0f))); }

// inverse of the height CDF
INLINE_FUNC float invC1(const float U) { return max(-1.0f, min(1.0f, 2.0f * U - 1.0f)); }

} // namespace mf_height_uniform

namespace mf_height_gaussian
{
// height PDF
INLINE_FUNC float P1(const float h) { return INV_SQRT_2PI * exp(-0.5f * h * h); }

// height CDF
INLINE_FUNC float C1(const float h) { return 0.5f + 0.5f * (float)mf_erf(INV_SQRT_2 * h); }

// inverse of the height CDF
INLINE_FUNC float invC1(const float U) { return SQRT_2 * mf_erfinv(2.0f * U - 1.0f); }
} // namespace mf_height_gaussian

namespace mf_slope_beckmann
{
// distribution of slopes
INLINE_FUNC float P22(const float alpha_x, const float alpha_y, const float slope_x, const float slope_y)
{
	const float value = 1.0f / (PI * alpha_x * alpha_y) *
						expf(-slope_x * slope_x / (alpha_x * alpha_x) - slope_y * slope_y / (alpha_y * alpha_y));
	return value;
}
// Smith's Lambda function
INLINE_FUNC float Lambda(const float alpha_x, const float alpha_y, const vec3 wi)
{
	if (wi.z > 0.9999f)
		return 0.0f;
	if (wi.z < -0.9999f)
		return -1.0f;

	// a
	const float theta_i = acos(wi.z);
	const float a = 1.0f / tan(theta_i) / mf_slope::alpha_i(alpha_x, alpha_y, wi);

	// value
	const float value = 0.5f * ((float)mf_erf(a) - 1.0f) + HALF_INV_SQRT_PI / a * exp(-a * a);

	return value;
}
// projected area towards incident direction
INLINE_FUNC float projectedArea(const float alpha_x, const float alpha_y, const vec3 wi)
{
	if (wi.z > 0.9999f)
		return 1.0f;
	if (wi.z < -0.9999f)
		return 0.0f;

	// a
	const float alphai = mf_slope::alpha_i(alpha_x, alpha_y, wi);
	const float theta_i = acos(wi.z);
	const float a = 1.0f / tan(theta_i) / alphai;

	// value
	const float value =
		0.5f * ((float)mf_erf(a) + 1.0f) * wi.z + HALF_INV_SQRT_PI * alphai * sin(theta_i) * exp(-a * a);

	return value;
}
// sample the distribution of visible slopes with alpha=1.0
INLINE_FUNC vec2 sampleP22_11(const float alpha_x, const float alpha_y, const float theta_i, const float U,
							  const float U_2)
{
	vec2 slope;

	if (theta_i < 0.0001f)
	{
		const float r = sqrtf(-logf(U));
		const float phi = 6.28318530718f * U_2;
		slope.x = r * cosf(phi);
		slope.y = r * sinf(phi);
		return slope;
	}

	// constant
	const float sin_theta_i = sinf(theta_i);
	const float cos_theta_i = cosf(theta_i);

	// slope associated to theta_i
	const float slope_i = cos_theta_i / sin_theta_i;

	// projected area
	const float a = cos_theta_i / sin_theta_i;
	const float projectedarea =
		0.5f * ((float)mf_erf(a) + 1.0f) * cos_theta_i + HALF_INV_SQRT_PI * sin_theta_i * exp(-a * a);
	if (projectedarea < 0.0001f || projectedarea != projectedarea)
		return vec2(0, 0);
	// VNDF normalization factor
	const float c = 1.0f / projectedarea;

	// search
	float erf_min = -0.9999f;
	float erf_max = max(erf_min, float(mf_erf(slope_i)));
	float erf_current = 0.5f * (erf_min + erf_max);

	while (erf_max - erf_min > 0.00001f)
	{
		if (!(erf_current >= erf_min && erf_current <= erf_max))
			erf_current = 0.5f * (erf_min + erf_max);

		// evaluate slope
		const float slope = mf_erfinv(erf_current);

		// CDF
		const float CDF = (slope >= slope_i) ? 1.0f
											 : c * (HALF_INV_SQRT_PI * sin_theta_i * exp(-slope * slope) +
													cos_theta_i * (0.5f + 0.5f * float(erf(slope))));
		const float diff = CDF - U;

		// test estimate
		if (abs(diff) < 0.00001f)
			break;

		// update bounds
		if (diff > 0.0f)
		{
			if (erf_max == erf_current)
				break;
			erf_max = erf_current;
		}
		else
		{
			if (erf_min == erf_current)
				break;
			erf_min = erf_current;
		}

		// update estimate
		const float derivative = 0.5f * c * cos_theta_i - 0.5f * c * sin_theta_i * slope;
		erf_current -= diff / derivative;
	}

	slope.x = mf_erfinv(min(erf_max, max(erf_min, erf_current)));
	slope.y = mf_erfinv(2.0f * U_2 - 1.0f);
	return slope;
}
} // namespace mf_slope_beckmann

namespace mf_slope
{
INLINE_FUNC float alpha_i(float alpha_x, float alpha_y, const vec3 wi)
{
	const float invSinTheta2 = 1.0f / (1.0f - wi.z * wi.z);
	const float cosPhi2 = wi.x * wi.x * invSinTheta2;
	const float sinPhi2 = wi.y * wi.y * invSinTheta2;
	const float alpha_i = sqrt(cosPhi2 * alpha_x * alpha_x + sinPhi2 * alpha_y * alpha_y);
	return alpha_i;
}
// distribution of normals (NDF)
INLINE_FUNC float D(const float alpha_x, const float alpha_y, const vec3 wm)
{
	if (wm.z <= 0.0f)
		return 0.0f;

	// slope of wm
	const float slope_x = -wm.x / wm.z;
	const float slope_y = -wm.y / wm.z;

	// value
	const float value = mf_slope_beckmann::P22(alpha_x, alpha_y, slope_x, slope_y) / (wm.z * wm.z * wm.z * wm.z);
	return value;
}

// distribution of visible normals (VNDF)
INLINE_FUNC float D_wi(const float alpha_x, const float alpha_y, const vec3 wi, const vec3 wm)
{
	if (wm.z <= 0.0f)
		return 0.0f;

	// normalization coefficient
	const float projectedarea = mf_slope_beckmann::projectedArea(alpha_x, alpha_y, wi);
	if (projectedarea == 0)
		return 0;
	const float c = 1.0f / projectedarea;

	// value
	const float value = c * max(0.0f, dot(wi, wm)) * D(alpha_x, alpha_y, wm);
	return value;
}

// sample the VNDF
INLINE_FUNC vec3 sampleD_wi(const float alpha_x, const float alpha_y, const vec3 wi, const float U1, const float U2)
{
	// stretch to match configuration with alpha=1.0
	const vec3 wi_11 = normalize(vec3(alpha_x * wi.x, alpha_y * wi.y, wi.z));

	// sample visible slope with alpha=1.0
	vec2 slope_11 = mf_slope_beckmann::sampleP22_11(alpha_x, alpha_y, acos(wi_11.z), U1, U2);

	// align with view direction
	const float phi = atan2(wi_11.y, wi_11.x);
	vec2 slope(cos(phi) * slope_11.x - sin(phi) * slope_11.y, sin(phi) * slope_11.x + cos(phi) * slope_11.y);

	// stretch back
	slope.x *= alpha_x;
	slope.y *= alpha_y;

	// if numerical instability
	if ((slope.x != slope.x) || !is_finite_number(slope.x))
	{
		if (wi.z > 0)
			return vec3(0.0f, 0.0f, 1.0f);
		else
			return normalize(vec3(wi.x, wi.y, 0.0f));
	}

	// compute normal
	const vec3 wm = normalize(vec3(-slope.x, -slope.y, 1.0f));
	return wm;
}
} // namespace mf_slope

namespace microsurface
{
// masking function
INLINE_FUNC float G_1(const float alpha_x, const float alpha_y, const vec3 wi)
{
	if (wi.z > 0.9999f)
		return 1.0f;
	if (wi.z <= 0.0f)
		return 0.0f;

	// Lambda
	const float Lambda = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi);
	// value
	const float value = 1.0f / (1.0f + Lambda);
	return value;
}
// masking function at height h0
INLINE_FUNC float G_1(const float alpha_x, const float alpha_y, const vec3 wi, const float h0)
{
	if (wi.z > 0.9999f)
		return 1.0f;
	if (wi.z <= 0.0f)
		return 0.0f;

	// height CDF
	const float C1_h0 = mf_height_uniform::C1(h0);
	// Lambda
	const float Lambda = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi);
	// value
	const float value = powf(C1_h0, Lambda);
	return value;
}
// sample height in outgoing direction
INLINE_FUNC float sampleHeight(const float alpha_x, const float alpha_y, const vec3 wr, const float hr, const float U)
{
	if (wr.z > 0.9999f)
		return FLT_MAX;
	if (wr.z < -0.9999f)
	{
		const float value = mf_height_uniform::invC1(U * mf_height_uniform::C1(hr));
		return value;
	}
	if (fabsf(wr.z) < 0.0001f)
		return hr;

	// probability of intersection
	const float G_1_ = G_1(alpha_x, alpha_y, wr, hr);

	if (U > 1.0f - G_1_) // leave the microsurface
		return FLT_MAX;

	const float h = mf_height_uniform::invC1(mf_height_uniform::C1(hr) /
											 pow((1.0f - U), 1.0f / mf_slope_beckmann::Lambda(alpha_x, alpha_y, wr)));
	return h;
}
} // namespace microsurface

namespace microsurf_conductor
{

// evaluate local phase function
INLINE_FUNC float evalPhaseFunction(const float alpha_x, const float alpha_y, const vec3 wi, const vec3 wo)
{
	// half vector
	const vec3 wh = normalize(wi + wo);
	if (wh.z < 0.0f)
		return 0.0f;

	// value
	const float value = 0.25f * mf_slope::D_wi(alpha_x, alpha_y, wi, wh) / dot(wi, wh);
	return value;
}

// sample local phase function
INLINE_FUNC vec3 samplePhaseFunction(const float alpha_x, const float alpha_y, float r1, float r2, const vec3 wi)
{
	const float U1 = r1;
	const float U2 = r2;

	vec3 wm = mf_slope::sampleD_wi(alpha_x, alpha_y, wi, U1, U2);

	// reflect
	const vec3 wo = -wi + 2.0f * wm * dot(wi, wm);

	return wo;
}

// evaluate BSDF limited to single scattering
// this is in average equivalent to eval(wi, wo, 1);
INLINE_FUNC float evalSingleScattering(const float alpha_x, const float alpha_y, const vec3 wi, const vec3 wo)
{
	// half-vector
	const vec3 wh = normalize(wi + wo);
	const float D = mf_slope::D(alpha_x, alpha_y, wh);

	// masking-shadowing
	const float G2 = 1.0f / (1.0f + mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi) +
							 mf_slope_beckmann::Lambda(alpha_x, alpha_y, wo));

	// BRDF * cos
	const float value = D * G2 / (4.0f * wi.z);

	return value;
}

// evaluate BSDF with a random walk (stochastic but unbiased)
// scatteringOrder=0 --> contribution from all scattering events
// scatteringOrder=1 --> contribution from 1st bounce only
// scatteringOrder=2 --> contribution from 2nd bounce only, etc..
INLINE_FUNC float eval(const float alpha_x, const float alpha_y, const float r1, const float r2, const float r3,
					   const vec3 wi, const vec3 wo, const int scatteringOrder = 0)
{
	if (wo.z < 0)
		return 0;
	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);

	float sum = 0;

	// random walk
	int current_scatteringOrder = 0;
	while (scatteringOrder == 0 || current_scatteringOrder <= scatteringOrder)
	{
		// next height
		float U = r1;
		hr = microsurface::sampleHeight(alpha_x, alpha_y, wr, hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX)
			break;
		else
			current_scatteringOrder++;

		// next event estimation
		float phasefunction = evalPhaseFunction(alpha_x, alpha_y, -wr, wo);
		float shadowing = microsurface::G_1(alpha_x, alpha_y, wo, hr);
		float I = phasefunction * shadowing;

		if (is_finite_number(I) && (scatteringOrder == 0 || current_scatteringOrder == scatteringOrder))
			sum += I;

		// next direction
		wr = samplePhaseFunction(alpha_x, alpha_y, r2, r3, -wr);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return 0.0f;
	}

	return sum;
}

// sample BSDF with a random walk
// scatteringOrder is set to the number of bounces computed for this sample
INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const vec3 wi, const float r1, const float r2,
						const float r3, int &scatteringOrder)
{
	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);

	// random walk
	scatteringOrder = 0;
	while (true)
	{
		// next height
		float U = r1;
		hr = microsurface::sampleHeight(alpha_x, alpha_y, wr, hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX)
			break;
		else
			scatteringOrder++;

		// next direction
		wr = samplePhaseFunction(alpha_x, alpha_y, r2, r3, -wr);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return vec3(0, 0, 1);
	}

	return wr;
}

INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const float r1, const float r2, const float r3,
						const vec3 wi)
{
	int scatteringOrder;
	return sample(alpha_x, alpha_y, wi, r1, r2, r3, scatteringOrder);
}
} // namespace microsurf_conductor

namespace microsurf_dielectric
{
INLINE_FUNC float fresnel(const vec3 wi, const vec3 wm, const float eta)
{
	const float cos_theta_i = dot(wi, wm);
	const float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);

	// total internal reflection
	if (cos_theta_t2 <= 0.0f)
		return 1.0f;

	const float cos_theta_t = sqrtf(cos_theta_t2);

	const float Rs = (cos_theta_i - eta * cos_theta_t) / (cos_theta_i + eta * cos_theta_t);
	const float Rp = (eta * cos_theta_i - cos_theta_t) / (eta * cos_theta_i + cos_theta_t);

	const float F = 0.5f * (Rs * Rs + Rp * Rp);
	return F;
}

INLINE_FUNC vec3 refract(const vec3 wi, const vec3 wm, const float eta)
{
	const float cos_theta_i = dot(wi, wm);
	const float cos_theta_t2 = 1.0f - (1.0f - cos_theta_i * cos_theta_i) / (eta * eta);
	const float cos_theta_t = -sqrt(max(0.0f, cos_theta_t2));

	return wm * (dot(wi, wm) / eta + cos_theta_t) - wi / eta;
}

INLINE_FUNC float evalPhaseFunction(const float alpha_x, const float alpha_y, float eta, const vec3 wi, const vec3 wo,
									const bool wi_outside, const bool wo_outside)
{
	eta = wi_outside ? eta : 1.0f / eta;

	if (wi_outside == wo_outside) // reflection
	{
		// half vector
		const vec3 wh = normalize(wi + wo);
		// value
		const float value =
			(wi_outside)
				? (0.25f * mf_slope::D_wi(alpha_x, alpha_y, wi, wh) / dot(wi, wh) * fresnel(wi, wh, eta))
				: (0.25f * mf_slope::D_wi(alpha_x, alpha_y, -wi, -wh) / dot(-wi, -wh) * fresnel(-wi, -wh, eta));
		return value;
	}
	else // transmission
	{
		vec3 wh = -normalize(wi + wo * eta);
		wh *= (wi_outside) ? (sign(wh.z)) : (-sign(wh.z));

		if (dot(wh, wi) < 0)
			return 0;

		float value;
		if (wi_outside)
		{
			value = eta * eta * (1.0f - fresnel(wi, wh, eta)) * mf_slope::D_wi(alpha_x, alpha_y, wi, wh) *
					max(0.0f, -dot(wo, wh)) * 1.0f / powf(dot(wi, wh) + eta * dot(wo, wh), 2.0f);
		}
		else
		{
			value = eta * eta * (1.0f - fresnel(-wi, -wh, eta)) * mf_slope::D_wi(alpha_x, alpha_y, -wi, -wh) *
					max(0.0f, -dot(-wo, -wh)) * 1.0f / powf(dot(-wi, -wh) + eta * dot(-wo, -wh), 2.0f);
		}

		return value;
	}
}

// evaluate local phase function
INLINE_FUNC float evalPhaseFunction(const float alpha_x, const float alpha_y, const float eta, const vec3 wi,
									const vec3 wo)
{
	return evalPhaseFunction(alpha_x, alpha_y, eta, wi, wo, true, true) +
		   evalPhaseFunction(alpha_x, alpha_y, eta, wi, wo, true, false);
}

INLINE_FUNC vec3 samplePhaseFunction(const vec3 wi, const float alpha_x, const float alpha_y, float eta, const float U1,
									 const float U2, const float r3, const bool wi_outside, bool &wo_outside)
{
	eta = wi_outside ? eta : 1.0f / eta;

	vec3 wm = wi_outside ? (mf_slope::sampleD_wi(alpha_x, alpha_y, wi, U1, U2))
						 : (-mf_slope::sampleD_wi(alpha_x, alpha_y, -wi, U1, U2));

	const float F = fresnel(wi, wm, eta);

	if (r3 < F)
	{
		const vec3 wo = -wi + 2.0f * wm * dot(wi, wm); // reflect
		return wo;
	}
	else
	{
		wo_outside = !wi_outside;
		const vec3 wo = refract(wi, wm, eta);
		return normalize(wo);
	}
}

// sample local phase function
INLINE_FUNC vec3 samplePhaseFunction(const float alpha_x, const float alpha_y, const float eta, const float r1,
									 const float r2, const float r3, const vec3 wi, REFERENCE_OF(bool) wo_outside)
{
	return samplePhaseFunction(wi, alpha_x, alpha_y, eta, r1, r2, r3, true, wo_outside);
}

// evaluate BSDF limited to single scattering
// this is in average equivalent to eval(wi, wo, 1);
INLINE_FUNC float evalSingleScattering(const float alpha_x, const float alpha_y, const float eta, const vec3 wi,
									   const vec3 wo)
{
	bool wi_outside = true;
	bool wo_outside = wo.z > 0;

	if (wo_outside) // reflection
	{
		// D
		const vec3 wh = normalize(vec3(wi + wo));
		const float D = mf_slope::D(alpha_x, alpha_y, wh);

		// masking shadowing
		const float Lambda_i = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi);
		const float Lambda_o = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wo);
		const float G2 = 1.0f / (1.0f + Lambda_i + Lambda_o);

		// BRDF
		const float value = fresnel(wi, wh, eta) * D * G2 / (4.0f * wi.z);
		return value;
	}
	else // refraction
	{
		// D
		vec3 wh = -normalize(wi + wo * eta);
		if (eta < 1.0f)
			wh = -wh;
		const float D = mf_slope::D(alpha_x, alpha_y, wh);

		// G2
		const float Lambda_i = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi);
		const float Lambda_o = mf_slope_beckmann::Lambda(alpha_x, alpha_y, -wo);
		const float G2 = (float)beta(1.0f + Lambda_i, 1.0f + Lambda_o);

		// BSDF
		const float value = max(0.0f, dot(wi, wh)) * max(0.0f, -dot(wo, wh)) * 1.0f / wi.z * eta * eta *
							(1.0f - fresnel(wi, wh, eta)) * G2 * D / pow(dot(wi, wh) + eta * dot(wo, wh), 2.0f);
		return value;
	}
}

// evaluate BSDF with a random walk (stochastic but unbiased)
// scatteringOrder=0 --> contribution from all scattering events
// scatteringOrder=1 --> contribution from 1st bounce only
// scatteringOrder=2 --> contribution from 2nd bounce only, etc..
INLINE_FUNC float eval(const float alpha_x, const float alpha_y, const float eta, const float r1, const float r2,
					   const float r3, const vec3 wi, const vec3 wo, bool &wo_outside, const int scatteringOrder = 0)
{
	// TODO: Might need more random numbers depending on scatteringorder

	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);
	bool outside = true;

	float sum = 0.0f;

	// random walk
	int current_scatteringOrder = 0;
	while (scatteringOrder == 0 || current_scatteringOrder <= scatteringOrder)
	{
		// next height
		float U = r1;
		hr = (outside) ? microsurface::sampleHeight(alpha_x, alpha_y, wr, hr, U)
					   : -microsurface::sampleHeight(alpha_x, alpha_y, -wr, -hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX || hr == -FLT_MAX)
			break;
		else
			current_scatteringOrder++;

		// next event estimation
		float phasefunction = evalPhaseFunction(alpha_x, alpha_y, eta, -wr, wo, outside, (wo.z > 0));
		float shadowing =
			(wo.z > 0) ? microsurface::G_1(alpha_x, alpha_y, wo, hr) : microsurface::G_1(alpha_x, alpha_y, -wo, -hr);
		float I = phasefunction * shadowing;

		if (is_finite_number(I) && (scatteringOrder == 0 || current_scatteringOrder == scatteringOrder))
			sum += I;

		// next direction
		wr = samplePhaseFunction(-wr, alpha_x, alpha_y, eta, r1, r2, r3, outside, wo_outside);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return 0.0f;
	}

	return sum;
}

// sample BSDF with a random walk
// scatteringOrder is set to the number of bounces computed for this sample
INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const float eta, const vec3 wi, const float r1,
						const float r2, const float r3, int &scatteringOrder)
{
	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);
	bool outside = true;

	// random walk
	scatteringOrder = 0;
	while (true)
	{
		// next height
		float U = r1;
		hr = (outside) ? microsurface::sampleHeight(alpha_x, alpha_y, wr, hr, U)
					   : -microsurface::sampleHeight(alpha_x, alpha_y, -wr, -hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX || hr == -FLT_MAX)
			break;
		else
			scatteringOrder++;

		// next direction
		wr = samplePhaseFunction(-wr, alpha_x, alpha_y, eta, r1, r2, r3, outside, outside);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return vec3(0, 0, 1);
	}

	return wr;
}

INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const float eta, const float r1, const float r2,
						const float r3, const vec3 wi)
{
	int scatteringOrder;
	return sample(alpha_x, alpha_y, eta, wi, r1, r2, r3, scatteringOrder);
}
} // namespace microsurf_dielectric

namespace microsurf_diffuse
{

// evaluate local phase function
INLINE_FUNC float evalPhaseFunction(const float alpha_x, const float alpha_y, const float r1, const float r2,
									const vec3 wi, const vec3 wo)
{
	const float U1 = r1;
	const float U2 = r2;
	vec3 wm = mf_slope::sampleD_wi(alpha_x, alpha_y, wi, U1, U2);

	// value
	const float value = 1.0f / PI * max(0.0f, dot(wo, wm));
	return value;
}

// sample local phase function
INLINE_FUNC vec3 samplePhaseFunction(const float alpha_x, const float alpha_y, const float U1, const float U2,
									 const const float U3, const float U4, const vec3 &wi)
{
	vec3 wm = mf_slope::sampleD_wi(alpha_x, alpha_y, wi, U1, U2);

	// sample diffuse reflection
	vec3 w1, w2;
	buildOrthonormalBasis(w1, w2, wm);

	float r1 = 2.0f * U3 - 1.0f;
	float r2 = 2.0f * U4 - 1.0f;

	// concentric map code from
	// http://psgraphics.blogspot.ch/2011/01/improved-code-for-concentric-map.html
	float phi, r;
	if (r1 == 0 && r2 == 0)
	{
		r = phi = 0;
	}
	else if (r1 * r1 > r2 * r2)
	{
		r = r1;
		phi = (PI / 4.0f) * (r2 / r1);
	}
	else
	{
		r = r2;
		phi = (PI / 2.0f) - (r1 / r2) * (PI / 4.0f);
	}
	float x = r * cos(phi);
	float y = r * sin(phi);
	float z = sqrt(max(0.0f, 1.0f - x * x - y * y));
	vec3 wo = x * w1 + y * w2 + z * wm;

	return wo;
}

// evaluate BSDF limited to single scattering
// this is in average equivalent to eval(wi, wo, 1);
INLINE_FUNC float evalSingleScattering(const float alpha_x, const float alpha_y, const float r1, const float r2,
									   const vec3 wi, const vec3 wo)
{
	// sample visible microfacet
	const float U1 = r1;
	const float U2 = r2;
	const vec3 wm = mf_slope::sampleD_wi(alpha_x, alpha_y, wi, U1, U2);

	// shadowing given masking
	const float Lambda_i = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wi);
	const float Lambda_o = mf_slope_beckmann::Lambda(alpha_x, alpha_y, wo);
	float G2_given_G1 = (1.0f + Lambda_i) / (1.0f + Lambda_i + Lambda_o);

	// evaluate diffuse and shadowing given masking
	const float value = 1.0f / PI * max(0.0f, dot(wm, wo)) * G2_given_G1;

	return value;
}
} // namespace microsurf_diffuse

namespace microsurface
{

INLINE_FUNC float eval_single(const float alpha_x, const float alpha_y, const float eta, const vec3 wi, const vec3 wo,
							  REFERENCE_OF(uint) seed)
{
	return microsurf_diffuse::evalSingleScattering(alpha_x, alpha_y, RandomFloat(seed), RandomFloat(seed), wi, wo);
}

// evaluate BSDF with a random walk (stochastic but unbiased)
// scatteringOrder=0 --> contribution from all scattering events
// scatteringOrder=1 --> contribution from 1st bounce only
// scatteringOrder=2 --> contribution from 2nd bounce only, etc..
INLINE_FUNC float eval(const float alpha_x, const float alpha_y, const float metallic, const float eta, const vec3 wi,
					   const vec3 wo, REFERENCE_OF(uint) seed, const int scatteringOrder = 0)
{
	if (wo.z < 0)
		return 0;
	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);

	float sum = 0;

	// random walk
	int current_scatteringOrder = 0;
	while (scatteringOrder == 0 || current_scatteringOrder <= scatteringOrder)
	{
		// next height
		float U = RandomFloat(seed);
		hr = sampleHeight(alpha_x, alpha_y, wr, hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX)
			break;
		else
			current_scatteringOrder++;

		// next event estimation
		float phasefunction =
			microsurf_diffuse::evalPhaseFunction(alpha_x, alpha_y, RandomFloat(seed), RandomFloat(seed), -wr, wo);
		float shadowing = microsurface::G_1(alpha_x, alpha_y, wo, hr);
		float I = phasefunction * shadowing;

		if (is_finite_number(I) && (scatteringOrder == 0 || current_scatteringOrder == scatteringOrder))
			sum += I;

		// next direction
		bool wo_outside;
		if (eta > 1.0f)
			wr = microsurf_dielectric::samplePhaseFunction(alpha_x, alpha_y, eta, RandomFloat(seed), RandomFloat(seed),
														   RandomFloat(seed), -wr, wo_outside);
		else if (metallic > 0.0f)
			wr = microsurf_conductor::samplePhaseFunction(alpha_x, alpha_y, RandomFloat(seed), RandomFloat(seed), -wr);
		else
			wr = microsurf_diffuse::samplePhaseFunction(alpha_x, alpha_y, RandomFloat(seed), RandomFloat(seed),
														RandomFloat(seed), RandomFloat(seed), -wr);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return 0.0f;
	}

	return sum;
}

// sample BSDF with a random walk
// scatteringOrder is set to the number of bounces computed for this sample
INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const float metallic, const float eta, const vec3 wi,
						REFERENCE_OF(int) scatteringOrder, float r0, float r1, float r2, float r3)
{
	// init
	vec3 wr = -wi;
	float hr = 1.0f + mf_height_uniform::invC1(0.999f);

	// random walk
	scatteringOrder = 0;
	while (true)
	{
		// next height
		float U = r0;
		hr = sampleHeight(alpha_x, alpha_y, wr, hr, U);

		// leave the microsurface?
		if (hr == FLT_MAX)
			break;
		else
			scatteringOrder++;

		// next direction

		bool wo_outside;
		if (eta > 1.0f)
			wr = microsurf_dielectric::samplePhaseFunction(alpha_x, alpha_y, eta, r1, r2, r3, -wr, wo_outside);
		else if (metallic > 0.0f)
			wr = microsurf_conductor::samplePhaseFunction(alpha_x, alpha_y, r1, r2, -wr);
		else
			wr = microsurf_diffuse::samplePhaseFunction(alpha_x, alpha_y, r0, r1, r2, r3, -wr);

		// if NaN (should not happen, just in case)
		if ((hr != hr) || (wr.z != wr.z))
			return vec3(0, 0, 1);
	}

	return wr;
}

INLINE_FUNC vec3 sample(const float alpha_x, const float alpha_y, const float metallic, const float eta, const vec3 wi,
						REFERENCE_OF(uint) seed, REFERENCE_OF(int) scatteringOrder)
{
	return sample(alpha_x, alpha_y, metallic, eta, seed, wi, scatteringOrder);
}
} // namespace microsurface

// To implement a bsdf, implement the following 2 functions:
INLINE_FUNC vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 B, const vec3 wi,
							  const vec3 wo, REFERENCE_OF(float) pdf)
{
	const float roughness = shadingData.getRoughness();

	pdf = microsurface::eval_single(roughness, roughness, shadingData.getEta(), worldToTangent(wi, iN, T, B),
									worldToTangent(wo, iN, T, B), seed);

	return shadingData.color;
}

INLINE_FUNC vec3 SampleBSDF(const ShadingData shadingData, const vec3 iN, const vec3 N, const vec3 T, const vec3 B,
							const vec3 wi, const float t, const bool backfacing, REFERENCE_OF(vec3) wo,
							REFERENCE_OF(float) pdf, , float r0, float r1, float r2, float r3)
{
	vec3 wiLocal = worldToTangent(wi, iN, T, B);

	const float roughness = shadingData.getRoughness();

	int scatteringOrder = 0;
	wo = microsurface::sample(roughness, roughness, shadingData.getMetallic(), shadingData.getEta(), seed, wiLocal,
							  scatteringOrder);
	pdf = microsurface::eval(roughness, roughness, shadingData.getMetallic(), shadingData.getEta(), wiLocal, wo, seed,
							 scatteringOrder);

	wo = normalize(worldToTangent(wo, T, B, iN));

	return shadingData.color;
}