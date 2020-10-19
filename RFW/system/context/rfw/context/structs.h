#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>
using namespace glm;
#include <rfw/context/settings.h>

#ifndef __CUDACC__
#include <half.hpp>
using namespace half_float;
#ifndef DEVICE_FUNC
#define DEVICE_FUNC inline
#endif
#else
#ifndef DEVICE_FUNC
#define DEVICE_FUNC __device__ inline
#endif
#include <cuda_fp16.h>
#endif

namespace rfw
{

struct Triangle
{
#ifndef __CUDACC__
	Triangle()
	{
		memset(this, 0, sizeof(Triangle));
		dummy1 = 1.0f;
		dummy2 = 1.0f;
		dummy3 = 1.0f;
	}
#endif
	float u0, u1, u2; // 12
#ifndef __CUDACC__
	int lightTriIdx = -1; // 16
#else
	int ltriIdx; // 4
#endif

	float v0, v1, v2; // 28
	uint material;	  // 32
	glm::vec3 vN0;	  // 44
	float Nx;		  // 48
	glm::vec3 vN1;	  // 60
	float Ny;		  // 64
	glm::vec3 vN2;	  // 76
	float Nz;		  // 80
	glm::vec3 T;	  // 92
	float area;		  // 96
	glm::vec3 B;	  // 108
	float LOD;		  // 112

	glm::vec3 vertex0; // 124
	float dummy1;	   // 128
	glm::vec3 vertex1; // 140
	float dummy2;	   // 144
	glm::vec3 vertex2; // 156
	float dummy3;	   // 160 bytes.

	static float calculateArea(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2);

	void updateArea();
};

enum MatPropFlags
{
	IsDielectric = 0,
	DiffuseMapIsHDR = 1,
	HasDiffuseMap = 2,
	HasNormalMap = 3,
	HasSpecularityMap = 4,
	HasRoughnessMap = 5,
	IsAnisotropic = 6,
	Has2ndNormalMap = 7,
	Has3rdNormalMap = 8,
	Has2ndDiffuseMap = 9,
	Has3rdDiffuseMap = 10,
	HasSmoothNormals = 11,
	HasAlpha = 12,
	HasAlphaMap = 13
};

struct Material
{
	// data to be read unconditionally
	half diffuse_r, diffuse_g, diffuse_b, transmittance_r, transmittance_g, transmittance_b;
	uint flags;

	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	glm::uvec4 parameters; // 16 Disney principled BRDF parameters, 0.8 fixed point
						   // texture / normal map descriptors; exactly 128-bit each

	/* read if bit  2 set */ short texwidth0, texheight0;
	half uscale0, vscale0, uoffs0, voffs0;
	uint texaddr0;
	/* read if bit 11 set */ short texwidth1, texheight1;
	half uscale1, vscale1, uoffs1, voffs1;
	uint texaddr1;
	/* read if bit 12 set */ short texwidth2, texheight2;
	half uscale2, vscale2, uoffs2, voffs2;
	uint texaddr2;
	/* read if bit  3 set */ short nmapwidth0, nmapheight0;
	half nuscale0, nvscale0, nuoffs0, nvoffs0;
	uint nmapaddr0;
	/* read if bit  9 set */ short nmapwidth1, nmapheight1;
	half nuscale1, nvscale1, nuoffs1, nvoffs1;
	uint nmapaddr1;
	/* read if bit 10 set */ short nmapwidth2, nmapheight2;
	half nuscale2, nvscale2, nuoffs2, nvoffs2;
	uint nmapaddr2;
	/* read if bit  4 set */ short smapwidth, smapheight;
	half suscale, svscale, suoffs, svoffs;
	uint smapaddr;
	/* read if bit  5 set */ short rmapwidth, rmapheight;
	half ruscale, rvscale, ruoffs, rvoffs;
	uint rmapaddr;
	/* read if bit 17 set */ short cmapwidth, cmapheight;
	half cuscale, cvscale, cuoffs, cvoffs;
	uint cmapaddr;
	/* read if bit 18 set */ short amapwidth, amapheight;
	half auscale, avscale, auoffs, avoffs;
	uint amapaddr;

#if defined(__CUDACC__) || defined(__NVCC__) || defined(WIN32) || defined(__linux__) || defined(__APPLE__)
#if defined(__CUDACC__) || defined(__NVCC__)
#define FLOAT_AS_UINT(x) (__float_as_uint((x)))
#else
#define FLOAT_AS_UINT(x) (*(uint *)&(x))
#endif
	DEVICE_FUNC static bool hasFlag(const uint flags, MatPropFlags flag) { return (flags & (1u << uint(flag))); }
	DEVICE_FUNC bool hasFlag(MatPropFlags flag) const { return (flags & (1u << uint(flag))); }
	DEVICE_FUNC glm::vec3 getColor() const { return glm::vec3(diffuse_r, diffuse_g, diffuse_b); }
	DEVICE_FUNC glm::vec3 getAbsorption() const { return glm::vec3(transmittance_r, transmittance_g, transmittance_b); }

	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	DEVICE_FUNC float getRoughness() const { return float((FLOAT_AS_UINT(parameters.x) >> 0u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getMetallic() const { return float((FLOAT_AS_UINT(parameters.x) >> 8u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getSpecTrans() const { return float((FLOAT_AS_UINT(parameters.x) >> 16u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getSpecularTint() const { return float((FLOAT_AS_UINT(parameters.x) >> 24u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getDiffuseTrans() const { return float((FLOAT_AS_UINT(parameters.y) >> 0u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getAnisotropic() const { return float((FLOAT_AS_UINT(parameters.y) >> 8u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getSheen() const { return float((FLOAT_AS_UINT(parameters.y) >> 16u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getSheenTint() const { return float((FLOAT_AS_UINT(parameters.y) >> 24u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getClearCoat() const { return float((FLOAT_AS_UINT(parameters.z) >> 0u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getClearCoatGloss() const { return float((FLOAT_AS_UINT(parameters.z) >> 8u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getScatterDist() const { return float((FLOAT_AS_UINT(parameters.z) >> 16u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getRelativeIOR() const { return float((FLOAT_AS_UINT(parameters.z) >> 24u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getFlatness() const { return float((FLOAT_AS_UINT(parameters.w) >> 0u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getIor() const { return float((FLOAT_AS_UINT(parameters.w) >> 8u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getCustom1() const { return float((FLOAT_AS_UINT(parameters.w) >> 16u) & 0xFFu) / 256.0f; }
	DEVICE_FUNC float getCustom2() const { return float((FLOAT_AS_UINT(parameters.w) >> 24u) & 0xFFu) / 256.0f; }
#endif
};

struct MaterialTexIds
{
	MaterialTexIds() { memset(texture, -1, 11 * sizeof(int)); }
	int texture[11]{};
};

enum TexelStorage
{
	RGBA32 = 0, // regular texture data, RenderCore::texel32data
	RGBA128 = 1 // hdr texture data, RenderCore::texel128data
};

struct Mesh
{
	// Per vertex data
	const glm::vec4 *vertices = nullptr;
	const glm::vec3 *normals = nullptr;
	const glm::vec2 *texCoords = nullptr;

	// Per face data
	const rfw::Triangle *triangles = nullptr;
	const glm::uvec3 *indices = nullptr;
	size_t vertexCount = 0;
	size_t triangleCount = 0;

	bool hasIndices() const { return indices != nullptr; }
	bool hasNormals() const { return normals != nullptr; }
	bool hasTexCoords() const { return texCoords != nullptr; }
};

struct TextureData
{
	enum DataType
	{
		FLOAT4,
		UINT
	};

	DataType type;
	uint width, height, texelCount;
	uint texAddr;
	void *data;
};

struct LightCount
{
	uint areaLightCount;
	uint pointLightCount;
	uint spotLightCount;
	uint directionalLightCount;
};

struct AreaLight
{
	glm::vec3 position;
	float energy;
	glm::vec3 normal;
	float area;
	glm::vec3 radiance;
	int dummy0;
	glm::vec3 vertex0;
	int triIdx;
	glm::vec3 vertex1;
	int instIdx;
	glm::vec3 vertex2;
	int dummy1;
};

struct PointLight
{
	glm::vec3 position;
	float energy;
	glm::vec3 radiance;
	int dummy;
};

struct SpotLight
{
	glm::vec3 position;
	float cosInner;
	glm::vec3 radiance;
	float cosOuter;
	glm::vec3 direction;
	float energy;
};

struct DirectionalLight
{
	glm::vec3 direction;
	float energy;
	glm::vec3 radiance;
	int dummy;
};

} // namespace rfw