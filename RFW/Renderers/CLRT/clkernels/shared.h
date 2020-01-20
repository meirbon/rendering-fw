#ifndef RENDERINGFW_RFW_RENDERERS_CLRT_CLKERNELS_SHARED_H
#define RENDERINGFW_RFW_RENDERERS_CLRT_CLKERNELS_SHARED_H

#define STAGE_PRIMARY_RAY 0
#define STAGE_SECONDARY_RAY 1
#define STAGE_SHADOW_RAY 2

#define EXTENSION_CNTR 0
#define SHADOW_CNTR 1

#if HOST_CODE
using float4 = glm::vec4;
using float3 = glm::vec3;
using float2 = glm::vec2;
using uint4 = glm::uvec4;
using uint3 = glm::uvec3;
using uint2 = glm::uvec2;
using int4 = glm::vec4;
using int3 = glm::vec3;
using int2 = glm::vec2;
#endif

typedef struct
{
	float4 Origin;
	float4 Direction;
	float4 Emission_pixelIdx;
} CLPotentialContribution;

typedef struct
{
	float4 pos_lensSize;
	float4 right_spreadAngle;
	float4 up;
	float4 p1;

	uint samplesTaken;
	float geometryEpsilon;
	int scrwidth;
	int scrheight;
} CLCamera;

typedef struct
{
	uint4 baseData4;
	uint4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	uint4 t0data4;
	uint4 t1data4;
	uint4 t2data4;
	uint4 n0data4;
	uint4 n1data4;
	uint4 n2data4;
	uint4 sdata4;
	uint4 rdata4;
	uint4 m0data4;
	uint4 m1data4;
} CLMaterial;

typedef struct
{
	float4 u4;	// w: light triangle idx
	float4 v4;	// w: material
	float4 vN0; // w: Nx
	float4 vN1; // w: Ny
	float4 vN2; // w: Nz
	float4 T;	// w: area
	float4 B;	// w: LOD
	float4 vertex0;
	float4 vertex1;
	float4 vertex2;
} CLTriangle;

#define CHAR2FLT(x, s) ((float(((x >> s) & 255u))) * (1.0f / 255.0f))

typedef struct
{
	float4 color;	   // w: flags
	float4 absorption; // w: area
	uint4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */

#define IS_SPECULAR (0)
#define IS_EMISSIVE (shadingData.color.x > 1.0f || shadingData.color.y > 1.0f || shadingData.color.z > 1.0f)
#define METALLIC CHAR2FLT(shadingData.parameters.x, 0)
#define SUBSURFACE CHAR2FLT(shadingData.parameters.x, 8)
#define SPECULAR CHAR2FLT(shadingData.parameters.x, 16)
#define ROUGHNESS (max(0.001f, CHAR2FLT(shadingData.parameters.x, 24)))
#define SPECTINT CHAR2FLT(shadingData.parameters.y, 0)
#define ANISOTROPIC CHAR2FLT(shadingData.parameters.y, 8)
#define SHEEN CHAR2FLT(shadingData.parameters.y, 16)
#define SHEENTINT CHAR2FLT(shadingData.parameters.y, 24)
#define CLEARCOAT CHAR2FLT(shadingData.parameters.z, 0)
#define CLEARCOATGLOSS CHAR2FLT(shadingData.parameters.z, 8)
#define TRANSMISSION CHAR2FLT(shadingData.parameters.z, 16)
#define ETA CHAR2FLT(shadingData.parameters.z, 24)
#define CUSTOM0 CHAR2FLT(shadingData.parameters.z, 24)
#define CUSTOM1 CHAR2FLT(shadingData.parameters.w, 0)
#define CUSTOM2 CHAR2FLT(shadingData.parameters.w, 8)
#define CUSTOM3 CHAR2FLT(shadingData.parameters.w, 16)
#define CUSTOM4 CHAR2FLT(shadingData.parameters.w, 24)
} CLShadingData;

typedef struct
{
	float4 position_energy;
	float4 normal_area;
	float4 radiance;
	float4 vertex0_triIdx;
	float4 vertex1_instIdx;
	float4 vertex2;
} CLDeviceAreaLight;

typedef struct
{
	float4 position_energy;
	float4 radiance;
} CLDevicePointLight;

typedef struct
{
	float4 position_cosInner;
	float4 radiance_cosOuter;
	float4 direction_energy;
} CLDeviceSpotLight;

typedef struct
{
	float4 direction_energy;
	float4 radiance;
} CLDeviceDirectionalLight;

#endif // RENDERINGFW_RFW_RENDERERS_CLRT_CLKERNELS_SHARED_H
