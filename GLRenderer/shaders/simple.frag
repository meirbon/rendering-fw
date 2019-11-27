#version 450

out vec4 Color;

in vec4 Pos;
in vec3 WPos;
in vec3 N;
in vec2 UV;

uniform vec3 ambient;

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
    vec3 direction;
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

void createTangentSpace(const vec3 N, inout vec3 T, inout vec3 B)
{
    const float sign = sign(N.z);
    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    T = vec3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = vec3(b, sign + N.y * N.y * a, -N.y);
}

vec3 tangentToWorld(const vec3 s, const vec3 N, const vec3 T, const vec3 B)
{
    return vec3(dot(T, s), dot(B, s), dot(N, s));
}

vec3 worldToTangent(const vec3 s, const vec3 N, const vec3 T, const vec3 B)
{
    return T * s.x + B * s.y + N * s.z;
}

void main()
{
    vec3 normal = N;
    const uint flags = floatBitsToUint(color_flags.w);
    vec3 finalColor = vec3(0);
    vec3 color = color_flags.xyz;

    if (!any(greaterThan(color, vec3(1))))
    {
        vec3 T, B;
        createTangentSpace(normal, T, B);
        if (MAT_HASDIFFUSEMAP)
        {
            const vec4 texel = texture(t0, UV);
            if (MAT_HASALPHA && texel.w < 0.5f)
            {
                discard;
            }

            color = color * texel.xyz;
            if (MAT_HAS2NDDIFFUSEMAP)// must have base texture; second and third layers are additive
            {
                color += texture(t1, UV).xyz;
            }
            if (MAT_HAS3RDDIFFUSEMAP)
            {
                color += texture(t2, UV).xyz;
            }
        }
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
            normal = normalize(normal * T + mapNormal * B + mapNormal * normal);
        }

        finalColor += color * ambient;

        for (int i = 0; i < AREA_LIGHT_COUNT; i++)
        {
            vec3 L = areaLights[i].position_area.xyz - WPos.xyz;
            const float dist2 = dot(L, L);
            L /= sqrt(dist2);
            const float NdotL = max(dot(normal, L), 0);
            const float LNdotL = max(-dot(areaLights[i].normal, L), 0);
            if (NdotL > 0 && LNdotL > 0)
            {
                finalColor += color * areaLights[i].radiance * NdotL * areaLights[i].position_area.w * (1.0f / dist2);
            }
        }
    }
    else
    {
        finalColor = color;
    }

    Color = vec4(finalColor, Pos.w);
}