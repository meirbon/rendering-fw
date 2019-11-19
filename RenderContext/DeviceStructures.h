#ifndef DEVICE_STRUCTURES_H
#define DEVICE_STRUCTURES_H

#ifdef __CUDACC__
#define GLM_FORCE_ALIGNED_GENTYPES
#endif

#include <glm/glm.hpp>
#include <glm/ext.hpp>

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
struct DeviceTriangle
{
	glm::vec4 u4;		 // w: light triangle idx
	glm::vec4 v4;		 // w: material
	glm::vec4 vN0;		 // w: Nx
	glm::vec4 vN1;		 // w: Ny
	glm::vec4 vN2;		 // w: Nz
	glm::vec4 T;		 // w: area
	glm::vec4 B;		 // w: LOD
	glm::vec4 vertex[3]; // 48

#ifdef __CUDACC__
	DEVICE_FUNC unsigned int getMatID() const { return __float_as_uint(v4.w); }
	DEVICE_FUNC int getLightTriangleIndex() const { return __float_as_int(v4.w); }
#else
	DEVICE_FUNC unsigned int getMatID() const { return *(unsigned int *)&v4[3]; }
	DEVICE_FUNC int getLightTriangleIndex() const { return *(int *)&v4[3]; }
#endif
	DEVICE_FUNC glm::vec3 getTexU() const { return glm::vec3(u4); }
	DEVICE_FUNC glm::vec3 getTexV() const { return glm::vec3(v4); }
	DEVICE_FUNC glm::vec3 getNormal() const { return glm::vec3(vN0[3], vN1[3], vN2[3]); }
	DEVICE_FUNC glm::vec3 getVertexN0() const { return glm::vec3(vN0); }
	DEVICE_FUNC glm::vec3 getVertexN1() const { return glm::vec3(vN1); }
	DEVICE_FUNC glm::vec3 getVertexN2() const { return glm::vec3(vN2); }
	DEVICE_FUNC float getArea() const { return T.w; }
	DEVICE_FUNC float getInverseArea() const { return 1.0f / getArea(); }
	DEVICE_FUNC float getLOD() const { return B.w; }
	DEVICE_FUNC glm::vec3 getVertex0() const { return glm::vec3(vertex[0]); }
	DEVICE_FUNC glm::vec3 getVertex1() const { return glm::vec3(vertex[1]); }
	DEVICE_FUNC glm::vec3 getVertex2() const { return glm::vec3(vertex[2]); }
};

struct DeviceMaterial
{
	glm::uvec4 baseData4;
	glm::uvec4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	glm::uvec4 t0data4;
	glm::uvec4 t1data4;
	glm::uvec4 t2data4;
	glm::uvec4 n0data4;
	glm::uvec4 n1data4;
	glm::uvec4 n2data4;
	glm::uvec4 sdata4;
	glm::uvec4 rdata4;
	glm::uvec4 m0data4;
	glm::uvec4 m1data4;
};

#define TEXTURE0 0
#define TEXTURE1 1
#define TEXTURE2 2
#define NORMALMAP0 3
#define NORMALMAP1 4
#define NORMALMAP2 5
#define SPECULARITY 6
#define ROUGHNESS0 7
#define ROUGHNESS1 8
#define COLORMASK 9
#define ALPHAMASK 10

struct DeviceInstanceDescriptor
{
	glm::mat3 invTransform;	// 3 * 3 * sizeof(float) =  36
	float dummy;			   // Fix alignment, 40
	DeviceTriangle *triangles; // 48
};

struct CameraView
{
	glm::vec3 pos = glm::vec3(0);
	glm::vec3 p1 = glm::vec3(-1, -1, -1);
	glm::vec3 p2 = glm::vec3(1, -1, -1);
	glm::vec3 p3 = glm::vec3(-1, 1, -1);
	float aperture = 0;
	float spreadAngle = 0.01f; // spread angle of center pixel
};

struct DeviceAreaLight
{
	glm::vec4 pos_energy;
	glm::vec4 normal_area;
	glm::vec4 radiance;
	glm::vec4 vertex0_triIdx;
	glm::vec4 vertex1_instIdx;
	glm::vec4 vertex2;

	DEVICE_FUNC glm::vec3 getPosition() const { return glm::vec3(pos_energy); }
	DEVICE_FUNC float getEnergy() const { return pos_energy.w; }
	DEVICE_FUNC glm::vec3 getNormal() const { return glm::vec3(normal_area); }
	DEVICE_FUNC glm::vec3 getRadiance() const { return glm::vec3(radiance); }
	DEVICE_FUNC float getArea() const { return normal_area.w; }
#ifdef __CUDACC__
	DEVICE_FUNC int getTriangleIdx() const { return __float_as_int(vertex0_triIdx.w); }
	DEVICE_FUNC int getInstanceIdx() const { return __float_as_int(vertex1_instIdx.w); }
#else
	DEVICE_FUNC int getTriangleIdx() const
	{
		int value;
		memcpy(&value, &vertex0_triIdx.w, sizeof(int));
		return value;
	}
	DEVICE_FUNC int getInstanceIdx() const
	{
		int value;
		memcpy(&value, &vertex1_instIdx.w, sizeof(int));
		return value;
	}
#endif
	DEVICE_FUNC glm::vec3 getVertex0() const { return glm::vec3(vertex0_triIdx); }
	DEVICE_FUNC glm::vec3 getVertex1() const { return glm::vec3(vertex1_instIdx); }
	DEVICE_FUNC glm::vec3 getVertex2() const { return glm::vec3(vertex2); }
};

struct DevicePointLight
{
	glm::vec4 position_energy;
	glm::vec4 radiance;

	DEVICE_FUNC glm::vec3 getPosition() const { return glm::vec3(position_energy); }
	DEVICE_FUNC float getEnergy() const { return position_energy.w; }
	DEVICE_FUNC glm::vec3 getRadiance() const { return glm::vec3(radiance); }
};

struct DeviceSpotLight
{
	glm::vec4 position_cosInner;
	glm::vec4 radiance_cosOuter;
	glm::vec4 direction_energy;

	DEVICE_FUNC glm::vec3 getPosition() const { return glm::vec3(position_cosInner); }
	DEVICE_FUNC glm::vec3 getRadiance() const { return glm::vec3(radiance_cosOuter); }
	DEVICE_FUNC glm::vec3 getDirection() const { return glm::vec3(direction_energy); }
	DEVICE_FUNC float getCosInner() const { return position_cosInner.w; }
	DEVICE_FUNC float getCosOuter() const { return radiance_cosOuter.w; }
	DEVICE_FUNC float getEnergy() const { return direction_energy.w; }
};

struct DeviceDirectionalLight
{
	glm::vec4 direction_energy;
	glm::vec4 radiance;

	DEVICE_FUNC glm::vec3 getDirection() const { return glm::vec3(direction_energy); }
	DEVICE_FUNC float getEnergy() const { return direction_energy.w; }
	DEVICE_FUNC glm::vec3 getRadiance() const { return glm::vec3(radiance); }
};

} // namespace rfw

#endif