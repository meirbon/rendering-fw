#ifndef GGXMDF_H
#define GGXMDF_H

#define BSDF_FUNC
#define REFERENCE_OF(x) inout x

#define DIFFWEIGHT weights.x
#define SHEENWEIGHT weights.y
#define SPECWEIGHT weights.z
#define COATWEIGHT weights.w

#define sqr(x) (x * x)

#define GGXMDF 1001
#define GTR1MDF 1002

BSDF_FUNC float stretched_roughness(const vec3 m, const float sin_theta, const float alpha_x, const float alpha_y)
{
	if (alpha_x == alpha_y || sin_theta == 0.0f)
		return 1.0f / sqr(alpha_x);
	const float cos_phi_2_ax_2 = sqr(m.x / (sin_theta * alpha_x));
	const float sin_phi_2_ay_2 = sqr(m.y / (sin_theta * alpha_y));
	return cos_phi_2_ax_2 + sin_phi_2_ay_2;
}

BSDF_FUNC float projected_roughness(const vec3 m, const float sin_theta, const float alpha_x, const float alpha_y)
{
	if (alpha_x == alpha_y || sin_theta == 0.0f)
		return alpha_x;
	const float cos_phi_2_ax_2 = sqr((m.x * alpha_x) / sin_theta);
	const float sin_phi_2_ay_2 = sqr((m.y * alpha_y) / sin_theta);
	return sqrt(cos_phi_2_ax_2 + sin_phi_2_ay_2);
}

BSDF_FUNC void sample_phi(const float s, REFERENCE_OF(float) cos_phi, REFERENCE_OF(float) sin_phi)
{
	const float phi = TWOPI * s;
	cos_phi = cos(phi);
	sin_phi = sin(phi);
}

BSDF_FUNC vec3 unit_vector(const float cos_theta, const float sin_theta, const float cos_phi, const float sin_phi)
{
	return vec3(cos_phi * sin_theta, sin_phi * sin_theta, cos_theta);
}

// AppleSeed note: Anisotropic GGX Microfacet Distribution Function.
// References:
//   [1] Microfacet Models for Refraction through Rough Surfaces
//       http://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf
//   [2] Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs
//       http://hal.inria.fr/docs/00/96/78/44/PDF/RR-8468.pdf
//   [3] Importance Sampling Microfacet-Based BSDFs using the Distribution of Visible Normals.
//       https://hal.inria.fr/hal-00996995/en
//   [4] A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals.
//       https://hal.archives-ouvertes.fr/hal-01509746

BSDF_FUNC float GGXMDF_D(const vec3 m, const float alpha_x, const float alpha_y)
{
	const float cos_theta = m.z;
	if (cos_theta == 0.0f)
		return (alpha_x * alpha_x) * INVPI;
	const float cos_theta_2 = (cos_theta * cos_theta);
	const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta_2));
	const float cos_theta_4 = (cos_theta_2 * cos_theta_2);
	const float tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2;
	const float A = stretched_roughness(m, sin_theta, alpha_x, alpha_y);
	const float tmp = 1.0f + tan_theta_2 * A;
	return 1.0f / (PI * alpha_x * alpha_y * cos_theta_4 * (tmp * tmp));
}

BSDF_FUNC float GGXMDF_lambda(const vec3 v, const float alpha_x, const float alpha_y)
{
	const float cos_theta = v.z;
	if (cos_theta == 0.0f)
		return 0.0f;
	const float cos_theta_2 = (cos_theta * cos_theta);
	const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta_2));
	const float alpha = projected_roughness(v, sin_theta, alpha_x, alpha_y);
	const float tan_theta_2 = sqr(sin_theta) / cos_theta_2;
	const float a2_rcp = (alpha * alpha) * tan_theta_2;
	return (-1.0f + sqrt(1.0f + a2_rcp)) * 0.5f;
}

BSDF_FUNC float GGXMDF_G(const vec3 wi, const vec3 wo, const vec3 m, const float alpha_x, const float alpha_y)
{
	return 1.0f / (1.0f + GGXMDF_lambda(wo, alpha_x, alpha_y) + GGXMDF_lambda(wi, alpha_x, alpha_y));
}

BSDF_FUNC float GGXMDF_G1(const vec3 v, const vec3 m, const float alpha_x, const float alpha_y) { return 1.0f / (1.0f + GGXMDF_lambda(v, alpha_x, alpha_y)); }

BSDF_FUNC vec3 GGXMDF_sample(const vec3 v, const float r0, const float r1, const float alpha_x, const float alpha_y)
{
	// stretch incident
	const float sign_cos_vn = v.z < 0.0f ? -1.0f : 1.0f;
	vec3 stretched = vec3(sign_cos_vn * v.x * alpha_x, sign_cos_vn * v.y * alpha_y, sign_cos_vn * v.z);
	stretched = normalize(stretched);

	// build an orthonormal basis
	const vec3 t1 = v.z < 0.9999f ? normalize(cross(stretched, vec3(0, 0, 1))) : vec3(1, 0, 0);
	const vec3 t2 = cross(t1, stretched);

	// sample point with polar coordinates (r, phi)
	const float a = 1.0f / (1.0f + stretched.z);
	const float r = sqrt(r0);
	const float phi = r1 < a ? r1 / a * PI : PI + (r1 - a) / (1.0f - a) * PI;
	const float p1 = r * cos(phi);
	const float p2 = r * sin(phi) * (r1 < a ? 1.0f : stretched.z);

	// compute normal
	const vec3 h = p1 * t1 + p2 * t2 + sqrt(max(0.0f, 1.0f - p1 * p1 - p2 * p2)) * stretched;

	// unstretch and normalize
	const vec3 m = vec3(h.x * alpha_x, h.y * alpha_y, max(0.0f, h.z));
	return normalize(m);
}

BSDF_FUNC float GGXMDF_D(const vec3 m, const float alpha)
{
	const float cos_theta = m.z;
	if (cos_theta == 0.0f)
		return (alpha * alpha) * INVPI;
	const float a2 = (alpha * alpha);
	const float cos_theta_2 = (cos_theta * cos_theta);
	const float cos_theta_4 = (cos_theta_2 * cos_theta_2);
	const float tan_theta_2 = (1.0f - cos_theta_2) / cos_theta_2;
	const float A = 1.0f / a2;
	const float tmp = 1.0f + tan_theta_2 * A;
	return 1.0f / (PI * a2 * cos_theta_4 * (tmp * tmp));
}

BSDF_FUNC float GGXMDF_lambda(const vec3 v, const float alpha)
{
	const float cos_theta = v.z;
	if (cos_theta == 0.0f)
		return 0;
	const float cos_theta_2 = (cos_theta * cos_theta);
	const float sin_theta = sqrt(max(0.0f, 1 - cos_theta_2));
	const float tan_theta_2 = (sin_theta * sin_theta) / cos_theta_2;
	const float a2_rcp = sqr(alpha) * tan_theta_2;
	return (-1.0f + sqrt(1.0f + a2_rcp)) * 0.5f;
}

BSDF_FUNC float GGXMDF_G(const vec3 wi, const vec3 wo, const vec3 m, const float alpha)
{
	return 1.0f / (1.0f + GGXMDF_lambda(wo, alpha) + GGXMDF_lambda(wi, alpha));
}

BSDF_FUNC float GGXMDF_G1(const vec3 v, const vec3 m, const float alpha) { return 1.0f / (1.0f + GGXMDF_lambda(v, alpha)); }

BSDF_FUNC vec3 GGXMDF_sample(const vec3 v, const float r0, const float r1, const float alpha) { return GGXMDF_sample(v, r0, r1, alpha, alpha); }

BSDF_FUNC float GGXMDF_pdf(const vec3 v, const vec3 m, const float alpha)
{
	const float cos_theta_v = v.z;
	if (cos_theta_v == 0.0f)
		return 0;
	return GGXMDF_G1(v, m, alpha) * abs(dot(v, m)) * GGXMDF_D(m, alpha) / abs(cos_theta_v);
}

BSDF_FUNC float pdf_visible_normals(const vec3 v, const vec3 m, const float alpha_x, const float alpha_y)
{
	const float cos_theta_v = v.z;
	if (cos_theta_v == 0.0f)
		return 0;
	return GGXMDF_G1(v, m, alpha_x, alpha_y) * abs(dot(v, m)) * GGXMDF_D(m, alpha_x, alpha_y) / abs(cos_theta_v);
}

BSDF_FUNC float GGXMDF_pdf(const vec3 v, const vec3 m, const float alpha_x, const float alpha_y) { return pdf_visible_normals(v, m, alpha_x, alpha_y); }

// AppleSeed note: GTR1 Microfacet Distribution Function.
// References:
//   [1] Physically-Based Shading at Disney
//       https://disney-animation.s3.amazonaws.com/library/s2012_pbs_disney_brdf_notes_v2.pdf
//   [2] Deriving the Smith shadowing function G1 for gamma (0, 4]
//       https://docs.chaosgroup.com/download/attachments/7147732/gtr_shadowing.pdf?version=2&modificationDate=1434539612000&api=v2

BSDF_FUNC float GTR1MDF_D(const vec3 m, const float alpha_x, const float alpha_y)
{
	const float alpha = clamp(alpha_x, 0.001f, 0.999f);
	const float alpha_x_2 = (alpha * alpha);
	const float cos_theta_2 = (m.z * m.z);
	const float a = (alpha_x_2 - 1.0f) / (PI * log(alpha_x_2));
	const float b = (1 / (1 + (alpha_x_2 - 1) * cos_theta_2));
	return a * b;
}

BSDF_FUNC float GTR1MDF_lambda(const vec3 v, const float alpha_x, const float alpha_y)
{
	const float cos_theta = v.z;
	if (cos_theta == 0)
		return 0;
	// [2] section 3.2
	const float cos_theta_2 = (cos_theta * cos_theta);
	const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta_2));
	// normal incidence; no shadowing
	if (sin_theta == 0.0f)
		return 0.0f;
	const float cot_theta_2 = cos_theta_2 / sqr(sin_theta);
	const float cot_theta = sqrt(cot_theta_2);
	const float alpha = clamp(alpha_x, 0.001f, 0.999f);
	const float alpha_2 = (alpha * alpha);
	const float a = sqrt(cot_theta_2 + alpha_2);
	const float b = sqrt(cot_theta_2 + 1.0f);
	const float c = log(cot_theta + b);
	const float d = log(cot_theta + a);
	return (a - b + cot_theta * (c - d)) / (cot_theta * log(alpha_2));
}

BSDF_FUNC float GTR1MDF_G(const vec3 wi, const vec3 wo, const vec3 m, const float alpha_x, const float alpha_y)
{
	return 1.0f / (1.0f + GTR1MDF_lambda(wo, alpha_x, alpha_y) + GTR1MDF_lambda(wi, alpha_x, alpha_y));
}

BSDF_FUNC float GTR1MDF_G1(const vec3 v, const vec3 m, const float alpha_x, const float alpha_y) { return 1.0f / (1.0f + GTR1MDF_lambda(v, alpha_x, alpha_y)); }

BSDF_FUNC vec3 GTR1MDF_sample(const float r0, const float r1, const float alpha_x, const float alpha_y)
{
	const float alpha = clamp(alpha_x, 0.001f, 0.999f);
	const float alpha_2 = (alpha * alpha);
	const float a = 1.0f - pow(alpha_2, 1.0f - r0);
	const float cos_theta_2 = a / (1.0f - alpha_2);
	const float cos_theta = sqrt(cos_theta_2);
	const float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta_2));
	float cos_phi, sin_phi;
	sample_phi(r1, cos_phi, sin_phi);
	return unit_vector(cos_theta, sin_theta, cos_phi, sin_phi);
}

BSDF_FUNC float GTR1MDF_pdf(const vec3 v, const vec3 m, const float alpha_x, const float alpha_y) { return GTR1MDF_D(m, alpha_x, alpha_y) * abs(m.z); }

#endif