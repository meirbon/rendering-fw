#ifndef STRUCTURES_GLSL
#define STRUCTURES_GLSL

// clang-format off
struct PotentialContribution
{
	vec4 Origin;
	vec4 Direction;
	vec4 Emission_pixelIdx;
};

struct CameraView
{
	vec4 pos_lensSize;
	vec4 right_spreadAngle;
	vec4 up;
	vec4 p1;
	uint samplesTaken;
	float geometryEpsilon;
	int scrwidth;
	int scrheight;
};

//struct Camera
//{
//	vec4 posLensSize;
//	vec4 right_aperture;
//	vec4 up_spreadAngle;
//	vec4 p1;
//	int samplesTaken, phase;
//	int scrwidth, scrheight;
//};

struct DeviceMaterial
{
	uvec4 baseData4;
	uvec4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	uvec4 t0data4;
	uvec4 t1data4;
	uvec4 t2data4;
	uvec4 n0data4;
	uvec4 n1data4;
	uvec4 n2data4;
	uvec4 sdata4;
	uvec4 rdata4;
	uvec4 m0data4;
	uvec4 m1data4;
	//IsDielectric = 0,
	//DiffuseMapIsHDR = 1,
	//HasDiffuseMap = 2,
	//HasNormalMap = 3,
	//HasSpecularityMap = 4,
	//HasRoughnessMap = 5,
	//IsAnisotropic = 6,
	//Has2ndNormalMap = 7,
	//Has3rdNormalMap = 8,
	//Has2ndDiffuseMap = 9,
	//Has3rdDiffuseMap = 10,
	//HasSmoothNormals = 11,
	//HasAlpha = 12,
	//HasAlphaMap = 13
#define MAT_FLAGS					(uint(baseData.w))
#define MAT_ISDIELECTRIC			((flags & (1u << 0u)  ) != 0)
#define MAT_DIFFUSEMAPISHDR			((flags & (1u << 1u)  ) != 0)
#define MAT_HASDIFFUSEMAP			((flags & (1u << 2u)  ) != 0)
#define MAT_HASNORMALMAP			((flags & (1u << 3u)  ) != 0)
#define MAT_HASSPECULARITYMAP		((flags & (1u << 4u)  ) != 0)
#define MAT_HASROUGHNESSMAP			((flags & (1u << 5u)  ) != 0)
#define MAT_ISANISOTROPIC			((flags & (1u << 6u)  ) != 0)
#define MAT_HAS2NDNORMALMAP			((flags & (1u << 7u)  ) != 0)
#define MAT_HAS3RDNORMALMAP			((flags & (1u << 8u)  ) != 0)
#define MAT_HAS2NDDIFFUSEMAP		((flags & (1u << 9u)  ) != 0)
#define MAT_HAS3RDDIFFUSEMAP		((flags & (1u << 10u) ) != 0)
#define MAT_HASSMOOTHNORMALS		((flags & (1u << 11u) ) != 0)
#define MAT_HASALPHA				((flags & (1u << 12u) ) != 0)
#define MAT_HASALPHAMAP				((flags & (1u << 13u) ) != 0)
};

struct DeviceTriangle
{
	vec4 u4;				// w: light tri idx			tdata0
	vec4 v4;				// w: material				tdata1
	vec4 vN0;				// w: Nx					tdata2
	vec4 vN1;				// w: Ny					tdata3
	vec4 vN2;				// w: Nz					tdata4
	vec4 area_invArea_LOD;
	vec4 vertex0;		    // 48						tdata7
	vec4 vertex1;			//							tdata8
	vec4 vertex2;			//							tdata9
};

#define CHAR2FLT(x, s) ((float( ((x >> s) & 255)) ) * (1.0f / 255.0f))

struct ShadingData
{
	// This structure is filled for an intersection point. It will contain the spatially varying material properties.
	vec3 color; int flags;
	vec3 absorption; int matID;
	uvec4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
#define IS_SPECULAR (0)
#define IS_EMISSIVE (shadingData.color.x > 1.0f || shadingData.color.y > 1.0f || shadingData.color.z > 1.0f)
#define METALLIC CHAR2FLT( shadingData.parameters.x, 0 )
#define SUBSURFACE CHAR2FLT( shadingData.parameters.x, 8 )
#define SPECULAR CHAR2FLT( shadingData.parameters.x, 16 )
#define ROUGHNESS (max( 0.001f, CHAR2FLT( shadingData.parameters.x, 24 ) ))
#define SPECTINT CHAR2FLT( shadingData.parameters.y, 0 )
#define ANISOTROPIC CHAR2FLT( shadingData.parameters.y, 8 )
#define SHEEN CHAR2FLT( shadingData.parameters.y, 16 )
#define SHEENTINT CHAR2FLT( shadingData.parameters.y, 24 )
#define CLEARCOAT CHAR2FLT( shadingData.parameters.z, 0 )
#define CLEARCOATGLOSS CHAR2FLT( shadingData.parameters.z, 8 )
#define TRANSMISSION CHAR2FLT( shadingData.parameters.z, 16 )
#define ETA CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM0 CHAR2FLT( shadingData.parameters.z, 24 )
#define CUSTOM1 CHAR2FLT( shadingData.parameters.w, 0 )
#define CUSTOM2 CHAR2FLT( shadingData.parameters.w, 8 )
#define CUSTOM3 CHAR2FLT( shadingData.parameters.w, 16 )
#define CUSTOM4 CHAR2FLT( shadingData.parameters.w, 24 )
};

struct DeviceAreaLight
{
	vec4 position_energy;
	vec4 normal_area;
	vec4 radiance;
	vec4 vertex0_triIdx;
	vec4 vertex1_instIdx;
	vec4 vertex2;
};

struct DevicePointLight
{
	vec4 position_energy;
	vec4 radiance;
};

struct DeviceSpotLight
{
	vec4 position_cosInner;
	vec4 radiance_cosOuter;
	vec4 direction_energy;
};

struct DeviceDirectionalLight
{
	vec4 direction_energy;
	vec4 radiance;
};

// clang-format on
#endif