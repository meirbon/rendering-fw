#ifndef MICROFACET_H
#define MICROFACET_H

#include "structures.glsl"

#define BSDF_FUNC
#define INLINE_FUNC
#define REFERENCE_OF(x) inout x

INLINE_FUNC float ErfInv(float x)
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

INLINE_FUNC float Erf(float x)
{
	// Save the sign of x
	int sign = 1;
	if (x < 0.0f)
		sign = -1;
	x = abs(x);

	// A&S formula 7.1.26
	const float t = 1.0f / (1.0f + 0.3275911f * x);
	const float y = 1.0f - (((((1.061405429f * t + -1.453152027f) * t) + 1.421413741f) * t + -0.284496736f) * t + 0.254829592f) * t * exp(-x * x);

	return sign * y;
}

INLINE_FUNC float CosTheta(const vec3 w) { return w.z; }

INLINE_FUNC float Cos2Theta(const vec3 w) { return w.z * w.z; }

INLINE_FUNC float AbsCosTheta(const vec3 w) { return abs(w.z); }

INLINE_FUNC float Sin2Theta(const vec3 w) { return max(0.f, 1.f - Cos2Theta(w)); }

INLINE_FUNC float SinTheta(const vec3 w) { return sqrt(Sin2Theta(w)); }

INLINE_FUNC float TanTheta(const vec3 w) { return SinTheta(w) / CosTheta(w); }

INLINE_FUNC float Tan2Theta(const vec3 w) { return Sin2Theta(w) / Cos2Theta(w); }

INLINE_FUNC float CosPhi(const vec3 w)
{
	float sinTheta = SinTheta(w);
	return (sinTheta == 0) ? 1 : clamp(w.x / sinTheta, -1.f, 1.f);
}

INLINE_FUNC float SinPhi(const vec3 w)
{
	const float sinTheta = SinTheta(w);
	return (sinTheta == 0.0f) ? 0 : clamp(w.y / sinTheta, -1.f, 1.f);
}

INLINE_FUNC float Cos2Phi(const vec3 w) { return CosPhi(w) * CosPhi(w); }

INLINE_FUNC float Sin2Phi(const vec3 w) { return SinPhi(w) * SinPhi(w); }

INLINE_FUNC float CosDPhi(const vec3 wa, const vec3 wb)
{
	return clamp((wa.x * wb.x + wa.y * wb.y) / sqrt((wa.x * wa.x + wa.y * wa.y) * (wb.x * wb.x + wb.y * wb.y)), -1.f, 1.f);
}

INLINE_FUNC float G1(float lambda_w) { return 1.0f / (1.0f + lambda_w); }

INLINE_FUNC float D(const vec3 wh, float alphay, float alphax)
{
	const float tan2Theta = Tan2Theta(wh);
	if ((2.0f * tan2Theta) == tan2Theta)
		return 0.f;

	const float cos4Theta = Cos2Theta(wh) * Cos2Theta(wh);

	return exp(-tan2Theta * (Cos2Phi(wh) / (alphax * alphax) + Sin2Phi(wh) / (alphay * alphay))) / (3.141592653589f * alphax * alphay * cos4Theta);
}

INLINE_FUNC vec3 EvalF(const vec3 col, const vec3 vl, const vec3 m)
{
	const float w = 1.0f - max(0.0f, dot(vl, m));
	return col + (vec3(1.0f) - col) * w * w * w * w * w;
}

INLINE_FUNC float EvalLambda(const vec3 vl, const vec2 alpha)
{
	// E. Heitz, Understanding the Masking-Shadowing Function in
	// Microfacet-Based BRDFs, 2014
	const float cosTheta2 = vl.z * vl.z;
	const float cosPhi2st = vl.x * alpha.x;
	const float sinPhi2st = vl.y * alpha.y;
	const float a2 = cosTheta2 / (cosPhi2st * cosPhi2st + sinPhi2st * sinPhi2st);

	return (-1.0f + sqrt(1.0f + 1.0f / a2)) * 0.5f;
}

INLINE_FUNC float EvalG1(const vec3 vl, const vec2 alpha) { return 1.0f / (EvalLambda(vl, alpha) + 1.0f); }

INLINE_FUNC float EvalG2(const vec3 il, const vec3 ol, const vec2 alpha) { return 1.0f / (EvalLambda(il, alpha) + EvalLambda(ol, alpha) + 1.0f); }

INLINE_FUNC float EvalD(const vec3 m, const vec2 alpha)
{
	const float cosTheta2 = m.z * m.z;
	const float exponent = ((m.x * m.x) / (alpha.x * alpha.x) + (m.y * m.y) / (alpha.y * alpha.y)) / cosTheta2;
	const float root = (1.0f + exponent) * cosTheta2;
	return 1.0f / (3.141592653589f * alpha.x * alpha.y * root * root);
}

INLINE_FUNC float EvaluateFresnelEta(const float eta, const vec3 ol, const vec3 m)
{
	if (eta <= 1.0f)
		return 0.0f;
	const float cosThetaI = max(0.0f, dot(ol, m));
	const float scale = 1.0f / eta;
	const float cosThetaTSqr = 1.0f - (1.0f - cosThetaI * cosThetaI) * (scale * scale);
	if (cosThetaTSqr <= 0.0f)
		return 1.0f;
	const float cosThetaT = sqrt(cosThetaTSqr);
	const float rs = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);
	const float rp = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
	return 0.5f * (rs * rs + rp * rp);
}

INLINE_FUNC float lambda_ggx(const vec3 wo, float alphaX, float alphaY)
{
	const float absTanThetaO = abs(TanTheta(wo));
	if (2.0f * absTanThetaO == absTanThetaO)
		return 0.0f;

	const float alpha = sqrt(Cos2Phi(wo) * alphaX * alphaX + Sin2Phi(wo) * alphaY * alphaY);
	const float alpha2Tan2Theta = alpha * absTanThetaO * alpha * absTanThetaO;
	return (-1.0f + sqrt(1.0f + alpha2Tan2Theta)) / 2.0f;
}

INLINE_FUNC float pdf_ggx(const vec3 wo, const vec3 wh, const vec3 wi, float alphaX, float alphaY) { return G1(lambda_ggx(wo, alphaX, alphaY)); }

INLINE_FUNC vec3 SampleMicrofacet(const vec3 ol, const vec2 alpha, const vec2 r)
{
	// E. Heitz, "Sampling the GGX Distribution of Visible Normals", 2018
	const vec3 v = normalize(vec3(alpha.x * ol.x, alpha.y * ol.y, ol.z));
	const vec3 t1 = normalize(cross(v, vec3(0.0f, 0.0f, 1.0f)));
	const vec3 t2 = cross(t1, v);
	const float s = 0.5f * (1.0f + v.z);
	const float sr = sqrt(r.x);
	const float phi = 2.0f * 3.141592653589f * r.y;
	const float p1 = sr * cos(phi);
	const float p2 = (1.0f - s) * sqrt(1.0f - p1 * p1) + s * (sr * sin(phi));
	const vec3 n = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * v;
	return normalize(vec3(alpha.x * n.x, alpha.y * n.y, max(0.0f, n.z)));
}

BSDF_FUNC vec3 BsdfPDF(const vec3 col, const vec3 m, const vec3 ol, const vec3 il, const vec2 alpha, REFERENCE_OF(float) pdf)
{
	const vec3 F = EvalF(col, ol, m);
	const float D = EvalD(m, alpha);
	const float G1 = EvalG1(ol, alpha);
	const float G2 = EvalG1(il, alpha);
	pdf = (G1 * max(0.0f, dot(ol, m)) * D * (1.0f / ol.z)) * (0.25f * (1.0f / dot(il, m)));
	return (F * G1 * G1 * D) * (1.0f / (4.0f * il.z * ol.z));
}

BSDF_FUNC vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 B, const vec3 wo, const vec3 wi, REFERENCE_OF(float) pdf)
{
	const float roughness = ROUGHNESS;
	if (roughness < 0.01f) // Use purely specular BRDF for roughness below threshold
	{
		pdf = 1.0f;
		return vec3(shadingData.color);
	}

	const vec3 woLocal = worldToTangent(wo, iN, T, B);
	const vec3 wmLocal = worldToTangent(iN, iN, T, B);
	const vec3 wiLocal = worldToTangent(wi, iN, T, B);

	pdf = pdf_ggx(woLocal, wmLocal, wiLocal, roughness, roughness);
	return vec3(shadingData.color);
}

BSDF_FUNC vec3 SampleBSDF(const ShadingData shadingData, vec3 iN, const vec3 N, const vec3 T, const vec3 B, const vec3 wi, const float t, const bool backfacing,
						  const float r3, const float r4, REFERENCE_OF(vec3) wo, REFERENCE_OF(float) pdf, REFERENCE_OF(bool) specular)
{
	const float roughness = ROUGHNESS;

	if (roughness < 0.01f) // Use purely specular BRDF for roughness below threshold
	{
		wo = reflect(-wi, iN);
		pdf = 1.0f;
		return vec3(shadingData.color);
	}

	const vec3 wiLocal = worldToTangent(wi, iN, T, B);
	vec3 s = SampleMicrofacet(wiLocal, vec2(roughness), vec2(r3, r4));
	const vec3 woLocal = reflect(-wiLocal, s);
	pdf = pdf_ggx(woLocal, s, wiLocal, roughness, roughness);
	wo = tangentToWorld(woLocal, iN, T, B);

	if (dot(wo, N) <= 0.0f)
		pdf = 0.0f;

	return vec3(shadingData.color);
}

#endif
