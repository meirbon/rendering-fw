#version 410

#define INVPI 0.31830988618
#define PI 3.14159265359

out vec4 Color;

in vec4 Pos;
in vec3 WPos;
in vec3 N;
in vec2 UV;

uniform vec3 ambient;
uniform vec3 forward;

// Material
uniform vec4 color_flags;
uniform uvec4 parameters;
uniform sampler2D t0;
uniform sampler2D t1;
uniform sampler2D t2;
uniform sampler2D n0;
uniform sampler2D n1;
uniform sampler2D n2;
uniform sampler2D s;
uniform sampler2D r;

#define CHAR2FLT(x, s) ((float(((x >> s) & 255u))) * (1.0f / 255.0f))
#define IS_EMISSIVE (color_flags.x > 1.0f || color_flags.y > 1.0f || color_flags.z > 1.0f)
#define METALLIC CHAR2FLT(parameters.x, 0)
#define SUBSURFACE CHAR2FLT(parameters.x, 8)
#define SPECULAR CHAR2FLT(parameters.x, 16)
#define ROUGHNESS (max(0.001f, CHAR2FLT(parameters.x, 24)))
#define SPECTINT CHAR2FLT(parameters.y, 0)
#define ANISOTROPIC CHAR2FLT(parameters.y, 8)
#define SHEEN CHAR2FLT(parameters.y, 16)
#define SHEENTINT CHAR2FLT(parameters.y, 24)
#define CLEARCOAT CHAR2FLT(parameters.z, 0)
#define CLEARCOATGLOSS CHAR2FLT(parameters.z, 8)
#define TRANSMISSION CHAR2FLT(parameters.z, 16)
#define DUMMY0 CHAR2FLT(parameters.z, 24)
#define ETA uintBitsToFloat(parameters.w)

#define MAT_FLAGS                    (uint(baseData.w))
#define MAT_ISDIELECTRIC            ((flags & (1u << 0u)) != 0)
#define MAT_DIFFUSEMAPISHDR            ((flags & (1u << 1u)) != 0)
#define MAT_HASDIFFUSEMAP            ((flags & (1u << 2u)) != 0)
#define MAT_HASNORMALMAP            ((flags & (1u << 3u)) != 0)
#define MAT_HASSPECULARITYMAP        ((flags & (1u << 4u)) != 0)
#define MAT_HASROUGHNESSMAP            ((flags & (1u << 5u)) != 0)
#define MAT_ISANISOTROPIC            ((flags & (1u << 6u)) != 0)
#define MAT_HAS2NDNORMALMAP            ((flags & (1u << 7u)) != 0)
#define MAT_HAS3RDNORMALMAP            ((flags & (1u << 8u)) != 0)
#define MAT_HAS2NDDIFFUSEMAP        ((flags & (1u << 9u)) != 0)
#define MAT_HAS3RDDIFFUSEMAP        ((flags & (1u << 10u)) != 0)
#define MAT_HASSMOOTHNORMALS        ((flags & (1u << 11u)) != 0)
#define MAT_HASALPHA                ((flags & (1u << 12u)) != 0)
#define MAT_HASALPHAMAP                ((flags & (1u << 13u)) != 0)

struct AreaLight
{
    vec4 position_area;
    vec3 normal;
    vec3 radiance;
    vec3 vertex0;
    vec3 vertex1;
    vec3 vertex2;
};

struct PointLight
{
    vec4 position_energy;
    vec3 radiance;
};

struct SpotLight
{
    vec4 position_cos_inner;
    vec4 radiance_cos_outer;
    vec4 direction_energy;
};

struct DirectionalLight
{
    vec4 direction_energy;
    vec3 radiance;
};

uniform uvec4 lightCount;
uniform AreaLight areaLights[32];
uniform PointLight pointLights[32];
uniform SpotLight spotLights[32];
uniform DirectionalLight dirLights[32];

#define AREA_LIGHT_COUNT lightCount.x
#define POINT_LIGHT_COUNT lightCount.y
#define SPOT_LIGHT_COUNT lightCount.z
#define DIR_LIGHT_COUNT lightCount.w

void createTangentSpace(vec3 N, inout vec3 T, inout vec3 B)
{
    float sign = sign(N.z);
    float a = -1.0f / (sign + N.z);
    float b = N.x * N.y * a;
    T = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = vec3(b, sign + N.y * N.y * a, -N.y);
}

vec3 tangentToWorld(vec3 s, vec3 N, vec3 T, vec3 B)
{
    return vec3(dot(T, s), dot(B, s), dot(N, s));
}

vec3 worldToTangent(vec3 s, vec3 N, vec3 T, vec3 B)
{
    return T * s.x + B * s.y + N * s.z;
}

float GTR1(float NDotH, float a)
{
    if (a >= 1)
    return INVPI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NDotH * NDotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

float GTR2(float NDotH, float a)
{
    float a2 = a * a;
    float t = 1.0f + (a2 - 1.0f) * NDotH * NDotH;
    return a2 / (PI * t * t);
}

float SmithGGX(const float NDotv, const float alphaG)
{
    float a = alphaG * alphaG;
    float b = NDotv * NDotv;
    return 1 / (NDotv + sqrt(a + b - a * b));
}

float Fr(const float VDotN, const float eio)
{
    float SinThetaT2 = (eio * eio) * (1.0f - VDotN * VDotN);
    if (SinThetaT2 > 1.0f)
    return 1.0f;// TIR
    float LDotN = sqrt(1.0f - SinThetaT2);
    // todo: reformulate to remove this division
    float eta = 1.0f / eio;
    float r1 = (VDotN - eta * LDotN) / (VDotN + eta * LDotN);
    float r2 = (LDotN - eta * VDotN) / (LDotN + eta * VDotN);
    return 0.5f * ((r1 * r1) + (r2 * r2));
}

float SchlickFresnel(const float u)
{
    float m = clamp(1.0f - u, 0.0f, 1.0f);
    return float(m * m) * (m * m) * m;
}

vec3 BSDFEval(const vec3 N, const vec3 wo, const vec3 wi)
{
    float NDotL = dot(N, wi);
    float NDotV = dot(N, wo);
    vec3 H = normalize(wi + wo);
    float NDotH = dot(N, H);
    float LDotH = dot(wi, H);
    vec3 Cdlin = color_flags.xyz;
    float Cdlum = .3f * Cdlin.x + .6f * Cdlin.y + .1f * Cdlin.z;// luminance approx.
    vec3 Ctint = Cdlum > 0.0f ? Cdlin / Cdlum : vec3(1.0f);// normalize lum. to isolate hue+sat
    vec3 Cspec0 = mix(SPECULAR * .08f * mix(vec3(1.0f), Ctint, SPECTINT), Cdlin, METALLIC);
	vec3 bsdf = vec3(0);
	vec3 brdf = vec3(0);
	if (TRANSMISSION > 0.0f)
	{
		// evaluate BSDF
		if (NDotL <= 0)
		{
		    // transmission Fresnel
		    float F = Fr(NDotV, ETA);
		    bsdf = vec3((1.0f - F) / abs(NDotL) * (1.0f - METALLIC) * TRANSMISSION);
		}
		else
		{
			// specular lobe
			float a = ROUGHNESS;
			float Ds = GTR2(NDotH, a);

			// Fresnel term with the microfacet normal
			float FH = Fr(LDotH, ETA);
			vec3 Fs = mix(Cspec0, vec3(1.0f), FH);
			float Gs = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);
			bsdf = (Gs * Ds) * Fs;
		}
	}
	if (TRANSMISSION < 1.0f)
	{
		// evaluate BRDF
		if (NDotL <= 0)
		{
    		if (SUBSURFACE > 0.0f)
			{
				// take sqrt to account for entry/exit of the ray through the medium
				// this ensures transmitted light corresponds to the diffuse model
				vec3 s = sqrt(color_flags.xyz);
				float FL = SchlickFresnel(abs(NDotL)), FV = SchlickFresnel(NDotV);
				float Fd = (1.0f - 0.5f * FL) * (1.0f - 0.5f * FV);
				brdf = INVPI * s * SUBSURFACE * Fd * (1.0f - METALLIC);
			}
		}
		else
		{
			// specular
			float a = ROUGHNESS;
			float Ds = GTR2(NDotH, a);

			// Fresnel term with the microfacet normal
			float FH = SchlickFresnel(LDotH);
			vec3 Fs = mix(Cspec0, vec3(1), FH);
			float Gs = SmithGGX(NDotV, a) * SmithGGX(NDotL, a);

			// Diffuse fresnel - go from 1 at normal incidence to .5 at grazing
			// and mix in diffuse retro-reflection based on roughness
			float FL = SchlickFresnel(NDotL), FV = SchlickFresnel(NDotV);
			float Fd90 = 0.5 + 2.0f * LDotH * LDotH * a;
			float Fd = mix(1.0f, Fd90, FL) * mix(1.0f, Fd90, FV);

			// clearcoat (ior = 1.5 -> F0 = 0.04)
			float Dr = GTR1(NDotH, mix(.1, .001, CLEARCOATGLOSS));
			float Fc = mix(.04f, 1.0f, FH);
			float Gr = SmithGGX(NDotL, .25) * SmithGGX(NDotV, .25);

			brdf = INVPI * Fd * Cdlin * (1.0f - METALLIC) * (1.0f - SUBSURFACE) + Gs * Fs * Ds + CLEARCOAT * Gr * Fc * Dr;
		}
	}

	vec3 final = mix(brdf, bsdf, TRANSMISSION);

	return final;
}


vec3 evalLighting(vec3 matColor, float roughness, vec3 N, vec3 T, vec3 B, vec3 wo, vec3 wi)
{
    return BSDFEval(N, wo, wi);
}

void main()
{
    vec3 normal = N;
    uint flags = floatBitsToUint(color_flags.w);
    vec3 finalColor = vec3(0);
    vec3 color = color_flags.xyz;
    vec3 T, B;
    createTangentSpace(normal, T, B);

    // Normal mapping
    if (MAT_HASNORMALMAP)
    {
        vec3 mapNormal = texture(n0, UV).xyz;
        if (MAT_HAS2NDNORMALMAP)
        {
            mapNormal += texture(n1, UV).xyz;
        }

        if (MAT_HAS3RDNORMALMAP)
        {
            mapNormal += texture(n2, UV).xyz;
        }

        mapNormal = normalize(mapNormal);
        normal = normalize(mapNormal.x * T + mapNormal.y * B + mapNormal.z * normal);
    }

    Color = vec4(normal, Pos.w);
}