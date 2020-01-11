/* disney2.h - License information:

   This code has been adapted from AppleSeed: https://appleseedhq.net
   The AppleSeed software is released under the MIT license.
   Copyright (c) 2014-2018 Esteban Tovagliari, The appleseedhq Organization.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   // https://github.com/appleseedhq/appleseed/blob/master/src/appleseed/renderer/modeling/bsdf/disneybrdf.cpp
*/

#ifndef DISNEY2_H
#define DISNEY2_H

#include "tools.glsl"
#include "ggxmdf.glsl"

#define BSDF_FUNC
#define REFERENCE_OF(x) inout x

#define DIFFWEIGHT weights.x
#define SHEENWEIGHT weights.y
#define SPECWEIGHT weights.z
#define COATWEIGHT weights.w

#define GGXMDF 1001
#define GTR1MDF 1002

#define saturate(x) max(0, min(1, x))
#define lerp(x, y, z) mix(x, y, z)

struct InputValues
{
	float sheen;
	float metallic;
	float specular;
	float clearcoat;
	float clearcoat_gloss;
	float roughness;
	float anisotropic;
	float subsurface;
	float sheen_tint;
	float specular_tint;
	vec4 tint_color;
	vec4 base_color__luminance;
};

BSDF_FUNC float schlick_fresnel(const float u)
{
	const float m = saturate(1.0f - u), m2 = (m * m), m4 = (m2 * m2);
	return m4 * m;
}

BSDF_FUNC void mix_spectra(const vec3 a, const vec3 b, const float t, REFERENCE_OF(vec3) result) { result = (1.0f - t) * a + t * b; }

BSDF_FUNC void mix_one_with_spectra(const vec3 b, const float t, REFERENCE_OF(vec3) result) { result = (1.0f - t) + t * b; }

BSDF_FUNC void mix_spectra_with_one(const vec3 a, const float t, REFERENCE_OF(vec3) result) { result = (1.0f - t) * a + t; }

BSDF_FUNC float microfacet_alpha_from_roughness(const float roughness) { return max(0.001f, roughness * roughness); }

BSDF_FUNC void microfacet_alpha_from_roughness(const float roughness, const float anisotropy, REFERENCE_OF(float) alpha_x, REFERENCE_OF(float) alpha_y)
{
	const float square_roughness = roughness * roughness;
	const float aspect = sqrt(1.0f + anisotropy * (anisotropy < 0 ? 0.9f : -0.9f));
	alpha_x = max(0.001f, square_roughness / aspect);
	alpha_y = max(0.001f, square_roughness * aspect);
}

BSDF_FUNC float clearcoat_roughness(const InputValues disney) { return mix(0.1f, 0.001f, disney.clearcoat_gloss); }

BSDF_FUNC void DisneySpecularFresnel(const InputValues disney, const vec3 o, const vec3 h, REFERENCE_OF(vec3) value)
{
	mix_one_with_spectra(disney.tint_color.xyz, disney.specular_tint, value);
	value *= disney.specular * 0.08f;
	mix_spectra(value, disney.base_color__luminance.xyz, disney.metallic, value);
	const float cos_oh = abs(dot(o, h));
	mix_spectra_with_one(value, schlick_fresnel(cos_oh), value);
}

BSDF_FUNC void DisneyClearcoatFresnel(const InputValues disney, const vec3 o, const vec3 h, REFERENCE_OF(vec3) value)
{
	const float cos_oh = abs(dot(o, h));
	value = vec3(mix(0.04f, 1.0f, schlick_fresnel(cos_oh)) * 0.25f * disney.clearcoat);
}

BSDF_FUNC bool force_above_surface(REFERENCE_OF(vec3) direction, const vec3 normal)
{
	const float Eps = 1.0e-4f;
	const float cos_theta = dot(direction, normal);
	const float correction = Eps - cos_theta;
	if (correction <= 0)
		return false;
	direction = normalize(direction + correction * normal);
	return true;
}

BSDF_FUNC vec3 linear_rgb_to_ciexyz(const vec3 rgb)
{
	return vec3(max(0.0f, 0.412453f * rgb.x + 0.357580f * rgb.y + 0.180423f * rgb.z), max(0.0f, 0.212671f * rgb.x + 0.715160f * rgb.y + 0.072169f * rgb.z),
				max(0.0f, 0.019334f * rgb.x + 0.119193f * rgb.y + 0.950227f * rgb.z));
}

BSDF_FUNC vec3 ciexyz_to_linear_rgb(const vec3 xyz)
{
	return vec3(max(0.0f, 3.240479f * xyz.x - 1.537150f * xyz.y - 0.498535f * xyz.z), max(0.0f, -0.969256f * xyz.x + 1.875991f * xyz.y + 0.041556f * xyz.z),
				max(0.0f, 0.055648f * xyz.x - 0.204043f * xyz.y + 1.057311f * xyz.z));
}

BSDF_FUNC void sample_mf(const uint MDF, const bool flip, const InputValues disney, const float r0, const float r1, const float alpha_x, const float alpha_y,
						 const vec3 iN, const vec3 wow,
						 /* OUT: */ REFERENCE_OF(vec3) wiw, REFERENCE_OF(float) pdf, REFERENCE_OF(vec3) value)
{
	vec3 T, B;
	createTangentSpace(iN, T, B);
	vec3 wo = worldToTangent(wow, iN, T, B); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	if (wo.z == 0)
		return;
	if (flip)
		wo.z = abs(wo.z);
	// compute the incoming direction by sampling the MDF
	vec3 m = MDF == GGXMDF ? GGXMDF_sample(wo, r0, r1, alpha_x, alpha_y) : GTR1MDF_sample(r0, r1, alpha_x, alpha_y);
	vec3 wi = reflect(wo * -1.0f, m);
	// force the outgoing direction to lie above the geometric surface.
	const vec3 ng = vec3(0, 0, 1); // TODO: this should be the geometric normal moved to a tangent space setup using the interpolated normal.
								   // local_geometry.m_shading_basis.transform_to_local( local_geometry.m_geometric_normal );
	if (force_above_surface(wi, ng))
		m = normalize(wo + wi);
	if (wi.z == 0)
		return;
	const float cos_oh = dot(wo, m);
	pdf = (MDF == GGXMDF ? GGXMDF_pdf(wo, m, alpha_x, alpha_y) : GTR1MDF_pdf(wo, m, alpha_x, alpha_y)) / abs(4.0f * cos_oh);
	/* assert( pdf >= 0 ); */
	if (pdf < 1.0e-6f)
		return; // skip samples with very low probability
	const float D = MDF == GGXMDF ? GGXMDF_D(m, alpha_x, alpha_y) : GTR1MDF_D(m, alpha_x, alpha_y);
	const float G = MDF == GGXMDF ? GGXMDF_G(wi, wo, m, alpha_x, alpha_y) : GTR1MDF_G(wi, wo, m, alpha_x, alpha_y);
	if (MDF == GGXMDF)
		DisneySpecularFresnel(disney, wo, m, value);
	else
		DisneyClearcoatFresnel(disney, wo, m, value);
	value *= D * G / abs(4.0f * wo.z * wi.z);
	wiw = normalize(tangentToWorld(wi, iN, T, B));
}

BSDF_FUNC float evaluate_mf(const uint MDF, const bool flip, const InputValues disney, const float alpha_x, const float alpha_y, const vec3 iN, const vec3 wow,
							const vec3 wiw, REFERENCE_OF(vec3) bsdf)
{
	vec3 T, B;
	createTangentSpace(iN, T, B);
	vec3 wo = worldToTangent(wow, iN, T, B); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	vec3 wi = worldToTangent(wiw, iN, T, B); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	if (wo.z == 0 || wi.z == 0)
		return 0;
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip)
		wo.z = abs(wo.z), wi.z = abs(wi.z);
	const vec3 m = normalize(wi + wo);
	const float cos_oh = dot(wo, m);
	if (cos_oh == 0)
		return 0;
	const float D = MDF == GGXMDF ? GGXMDF_D(m, alpha_x, alpha_y) : GTR1MDF_D(m, alpha_x, alpha_y);
	const float G = MDF == GGXMDF ? GGXMDF_G(wi, wo, m, alpha_x, alpha_y) : GTR1MDF_G(wi, wo, m, alpha_x, alpha_y);
	if (MDF == GGXMDF)
		DisneySpecularFresnel(disney, wo, m, bsdf);
	else
		DisneyClearcoatFresnel(disney, wo, m, bsdf);
	bsdf *= D * G / abs(4.0f * wo.z * wi.z);
	return (MDF == GGXMDF ? GGXMDF_pdf(wo, m, alpha_x, alpha_y) : GTR1MDF_pdf(wo, m, alpha_x, alpha_y)) / abs(4.0f * cos_oh);
}

BSDF_FUNC float pdf_mf(const uint MDF, const bool flip, const float alpha_x, const float alpha_y, const vec3 iN, const vec3 wow, const vec3 wiw)
{
	vec3 T, B;
	createTangentSpace(iN, T, B);
	vec3 wo = worldToTangent(wow, iN, T, B); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	vec3 wi = worldToTangent(wiw, iN, T, B); // local_geometry.m_shading_basis.transform_to_local( outgoing );
	// flip the incoming and outgoing vectors to be in the same hemisphere as the shading normal if needed.
	if (flip)
		wo.z = abs(wo.z), wi.z = abs(wi.z);
	const vec3 m = normalize(wi + wo);
	const float cos_oh = dot(wo, m);
	if (cos_oh == 0)
		return 0;
	return (MDF == GGXMDF ? GGXMDF_pdf(wo, m, alpha_x, alpha_y) : GTR1MDF_pdf(wo, m, alpha_x, alpha_y)) / abs(4.0f * cos_oh);
}

BSDF_FUNC float evaluate_diffuse(const InputValues disney, const vec3 iN, const vec3 wow, const vec3 wiw, REFERENCE_OF(vec3) value)
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	// const vec3 n( local_geometry.m_shading_basis.get_normal() );
	const vec3 n = iN;
	const vec3 h = (normalize(wiw + wow));
	// using the absolute values of cos_on and cos_in creates discontinuities
	const float cos_on = dot(n, wow);
	const float cos_in = dot(n, wiw);
	const float cos_ih = dot(wiw, h);
	const float fl = schlick_fresnel(cos_in);
	const float fv = schlick_fresnel(cos_on);
	float fd = 0;
	if (disney.subsurface != 1.0f)
	{
		const float fd90 = 0.5f + 2.0f * (cos_ih * cos_ih) * disney.roughness;
		fd = mix(1.0f, fd90, fl) * mix(1.0f, fd90, fv);
	}
	if (disney.subsurface > 0)
	{
		// based on Hanrahan-Krueger BRDF approximation of isotropic BSRDF.
		// the 1.25 scale is used to (roughly) preserve albedo.
		// Fss90 is used to "flatten" retroreflection based on roughness.
		const float fss90 = (cos_ih * cos_ih) * disney.roughness;
		const float fss = mix(1.0f, fss90, fl) * mix(1.0f, fss90, fv);
		const float ss = 1.25f * (fss * (1.0f / (abs(cos_on) + abs(cos_in)) - 0.5f) + 0.5f);
		fd = mix(fd, ss, disney.subsurface);
	}
	value = disney.base_color__luminance.xyz * fd * INVPI * (1.0f - disney.metallic);
	return abs(cos_in) * INVPI;
}

BSDF_FUNC void sample_diffuse(const InputValues disney, const float r0, const float r1, const vec3 iN, const vec3 wow,
							  /* OUT: */ REFERENCE_OF(vec3) wiw, REFERENCE_OF(float) pdf, REFERENCE_OF(vec3) value)
{
	// compute the incoming direction
	const vec3 wi = DiffuseReflectionCosWeighted(r0, r1);
	vec3 T, B;
	createTangentSpace(iN, T, B);
	wiw = normalize(tangentToWorld(wi, iN, T, B));
	// compute the component value and the probability density of the sampled direction.
	pdf = evaluate_diffuse(disney, iN, wow, wiw, value);
	/* assert( pdf > 0 ); */
	if (pdf < 1.0e-6f)
		return;
}

BSDF_FUNC float evaluate_sheen(const InputValues disney, const vec3 wow, const vec3 wiw, REFERENCE_OF(vec3) value)
{
	// this code is mostly ported from the GLSL implementation in Disney's BRDF explorer.
	const vec3 h = (normalize(wow + wow));
	const float cos_ih = dot(wiw, h);
	const float fh = schlick_fresnel(cos_ih);
	mix_one_with_spectra(disney.tint_color.xyz, disney.sheen_tint, value);
	value *= fh * disney.sheen * (1.0f - disney.metallic);
	return 1.0f / (2 * PI); // return the probability density of the sampled direction
}

BSDF_FUNC void sample_sheen(const InputValues disney, const float r0, const float r1, const vec3 iN, const vec3 wow,
							/* OUT: */ REFERENCE_OF(vec3) wiw, REFERENCE_OF(float) pdf, REFERENCE_OF(vec3) value)
{
	// compute the incoming direction
	const vec3 wi = DiffuseReflectionCosWeighted(r0, r1);
	vec3 T, B;
	createTangentSpace(iN, T, B);
	wiw = normalize(tangentToWorld(wi, iN, T, B));
	// compute the component value and the probability density of the sampled direction
	pdf = evaluate_sheen(disney, wow, wiw, value);
	/* assert( pdf > 0 ); */
	if (pdf < 1.0e-6f)
		return;
}

BSDF_FUNC void sample_disney(const InputValues disney, const float r0, const float r1, const vec3 iN, const vec3 wow,
							 /* OUT: */ REFERENCE_OF(vec3) wiw, REFERENCE_OF(float) pdf, REFERENCE_OF(vec3) value)
{
	// compute component weights and cdf
	vec4 weights = vec4(lerp(disney.base_color__luminance.w, 0, disney.metallic), lerp(disney.sheen, 0, disney.metallic),
						lerp(disney.specular, 1, disney.metallic), disney.clearcoat * 0.25f);
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	const vec4 cdf = vec4(weights.x, weights.x + weights.y, weights.x + weights.y + weights.z, 0);
	// sample a random component
	float probability, component_pdf;
	vec3 contrib;
	if (r0 < cdf.x)
	{
		const float r2 = r0 / cdf.x; // reuse r0 after normalization
		sample_diffuse(disney, r2, r1, iN, wow, wiw, component_pdf, value);
		probability = DIFFWEIGHT * component_pdf, DIFFWEIGHT = 0;
	}
	else if (r0 < cdf.y)
	{
		const float r2 = (r0 - cdf.x) / (cdf.y - cdf.x); // reuse r0 after normalization
		sample_sheen(disney, r2, r1, iN, wow, wiw, component_pdf, value);
		probability = SHEENWEIGHT * component_pdf, SHEENWEIGHT = 0;
	}
	else if (r0 < cdf.z)
	{
		const float r2 = (r0 - cdf.y) / (cdf.z - cdf.y); // reuse r0 after normalization
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness(disney.roughness, disney.anisotropic, alpha_x, alpha_y);
		sample_mf(GGXMDF, false, disney, r2, r1, alpha_x, alpha_y, iN, wow, wiw, component_pdf, value);
		probability = SPECWEIGHT * component_pdf, SPECWEIGHT = 0;
	}
	else
	{
		const float r2 = (r0 - cdf.z) / (1 - cdf.z); // reuse r0 after normalization
		const float alpha = clearcoat_roughness(disney);
		sample_mf(GTR1MDF, false, disney, r2, r1, alpha, alpha, iN, wow, wiw, component_pdf, value);
		probability = COATWEIGHT * component_pdf, COATWEIGHT = 0;
	}
	if (DIFFWEIGHT > 0)
		probability += DIFFWEIGHT * evaluate_diffuse(disney, iN, wow, wiw, contrib), value += contrib;
	if (SHEENWEIGHT > 0)
		probability += SHEENWEIGHT * evaluate_sheen(disney, wow, wiw, contrib), value += contrib;
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness(disney.roughness, disney.anisotropic, alpha_x, alpha_y);
		probability += SPECWEIGHT * evaluate_mf(GGXMDF, false, disney, alpha_x, alpha_y, iN, wow, wiw, contrib);
		value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness(disney);
		probability += COATWEIGHT * evaluate_mf(GTR1MDF, false, disney, alpha, alpha, iN, wow, wiw, contrib);
		value += contrib;
	}
	if (probability > 1.0e-6f)
		pdf = probability;
	else
		pdf = 0;
}

BSDF_FUNC float evaluate_disney(const InputValues disney, const vec3 iN, const vec3 wow, const vec3 wiw, REFERENCE_OF(vec3) value)
{
	// compute component weights
	vec4 weights = vec4(lerp(disney.base_color__luminance.w, 0, disney.metallic), lerp(disney.sheen, 0, disney.metallic),
						lerp(disney.specular, 1, disney.metallic), disney.clearcoat * 0.25f);
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	float pdf = 0;
	value = vec3(0);
	if (DIFFWEIGHT > 0)
		pdf += DIFFWEIGHT * evaluate_diffuse(disney, iN, wow, wiw, value);
	if (SHEENWEIGHT > 0)
		pdf += SHEENWEIGHT * evaluate_sheen(disney, wow, wiw, value);
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness(disney.roughness, disney.anisotropic, alpha_x, alpha_y);
		vec3 contrib;
		const float spec_pdf = evaluate_mf(GGXMDF, false, disney, alpha_x, alpha_y, iN, wow, wiw, contrib);
		if (spec_pdf > 0)
			pdf += SPECWEIGHT * spec_pdf, value += contrib;
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness(disney);
		vec3 contrib;
		const float clearcoat_pdf = evaluate_mf(GTR1MDF, false, disney, alpha, alpha, iN, wow, wiw, contrib);
		if (clearcoat_pdf > 0)
			pdf += COATWEIGHT * clearcoat_pdf, value += contrib;
	}
	/* assert( pdf >= 0 ); */ return pdf;
}

BSDF_FUNC float evaluate_pdf(const InputValues disney, const vec3 iN, const vec3 wow, const vec3 wiw)
{
	// compute component weights
	vec4 weights = vec4(lerp(disney.base_color__luminance.w, 0, disney.metallic), lerp(disney.sheen, 0, disney.metallic),
						lerp(disney.specular, 1, disney.metallic), disney.clearcoat * 0.25f);
	weights *= 1.0f / (weights.x + weights.y + weights.z + weights.w);
	// compute pdf
	float pdf = 0;
	if (DIFFWEIGHT > 0)
		pdf += DIFFWEIGHT * abs(dot(wiw, iN)) * INVPI;
	if (SHEENWEIGHT > 0)
		pdf += SHEENWEIGHT * (1.0f / (2 * PI));
	if (SPECWEIGHT > 0)
	{
		float alpha_x, alpha_y;
		microfacet_alpha_from_roughness(disney.roughness, disney.anisotropic, alpha_x, alpha_y);
		pdf += SPECWEIGHT * pdf_mf(GGXMDF, false, alpha_x, alpha_y, iN, wow, wiw);
	}
	if (COATWEIGHT > 0)
	{
		const float alpha = clearcoat_roughness(disney);
		pdf += COATWEIGHT * pdf_mf(GTR1MDF, false, alpha, alpha, iN, wow, wiw);
	}
	/* assert( pdf >= 0 ); */ return pdf;
}

// ============================================
// CONVERSION
// ============================================

BSDF_FUNC vec3 EvaluateBSDF(const ShadingData shadingData, const vec3 iN, const vec3 T, const vec3 B, const vec3 wo, const vec3 wi, REFERENCE_OF(float) pdf)
{
	InputValues disney;
	disney.sheen = SHEEN;
	disney.metallic = METALLIC;
	disney.specular = SPECULAR;
	disney.clearcoat = CLEARCOAT;
	disney.clearcoat_gloss = CLEARCOATGLOSS;
	disney.roughness = ROUGHNESS;
	disney.anisotropic = ANISOTROPIC;
	disney.subsurface = SUBSURFACE;
	disney.sheen_tint = SHEENTINT;
	disney.specular_tint = SPECTINT;
	disney.base_color__luminance = vec4(shadingData.color.xyz, 0);
	const vec3 tint_xyz = linear_rgb_to_ciexyz(disney.base_color__luminance.xyz);
	disney.tint_color.xyz = tint_xyz.y > 0 ? ciexyz_to_linear_rgb(tint_xyz * (1.0f / tint_xyz.y)) : vec3(1);
	disney.base_color__luminance.w = tint_xyz.y;

	vec3 value;
	pdf = evaluate_disney(disney, iN, wo, wi, value);
	return value;
}

BSDF_FUNC vec3 SampleBSDF(const ShadingData shadingData, vec3 iN, const vec3 N, const vec3 T, const vec3 B, const vec3 wo, const float t, const bool backfacing,
						  const float r3, const float r4, REFERENCE_OF(vec3) wi, REFERENCE_OF(float) pdf, REFERENCE_OF(bool) specular)
{
	InputValues disney;
	disney.sheen = SHEEN;
	disney.metallic = METALLIC;
	disney.specular = SPECULAR;
	disney.clearcoat = CLEARCOAT;
	disney.clearcoat_gloss = CLEARCOATGLOSS;
	disney.roughness = ROUGHNESS;
	disney.anisotropic = ANISOTROPIC;
	disney.subsurface = SUBSURFACE;
	disney.sheen_tint = SHEENTINT;
	disney.specular_tint = SPECTINT;
	disney.base_color__luminance = shadingData.color;

	const vec3 tint_xyz = linear_rgb_to_ciexyz(disney.base_color__luminance.xyz);
	disney.tint_color.xyz = tint_xyz.y > 0 ? ciexyz_to_linear_rgb(tint_xyz * (1.0f / tint_xyz.y)) : vec3(1);
	disney.base_color__luminance.w = tint_xyz.y;

	vec3 value;
	sample_disney(disney, r3, r4, iN, wo, wi, pdf, value);
	return value;
}

#endif