#include "Shared.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "BVH/BVHNode.h"
#include "BVH/MBVHNode.h"

#include "CUDAIntersect.h"

#include "getShadingData.h"

#include "bsdf.h"
#include "lights.h"

#define USE_TOP_MBVH 1
#define USE_MBVH 1
#define IS_SPECULAR 1
#define IS_LIGHTS 1
#define MAX_IS_LIGHTS 16

#define NEXTMULTIPLEOF(a, b) (((a) + ((b)-1)) & (0x7fffffff - ((b)-1)))
using namespace glm;

#ifndef __launch_bounds__ // Fix errors in IDE
void __sincosf(float, float *, float *) {}

#define __launch_bounds__(x, y)
int __float_as_int(float x) { return int(x); }
uint __float_as_uint(float x) { return uint(x); }
float __uint_as_float(uint x) { return float(x); }
float __int_as_float(int x) { return float(x); }

template <typename T, typename B> T atomicAdd(T *, B) { return T; }

template <typename T, int x> struct surface
{
};
template <typename T> void surf2Dwrite(T value, surface<void, cudaSurfaceType2D> output, size_t stride, size_t y, cudaSurfaceBoundaryMode mode) {}
#endif

surface<void, cudaSurfaceType2D> output;

__constant__ __device__ float geometryEpsilon;
__constant__ __device__ CameraView *view;
__constant__ __device__ Counters *counters;
__constant__ __device__ glm::vec4 *accumulator;
__constant__ __device__ uint stride;
__constant__ __device__ glm::vec4 *pathStates;
__constant__ __device__ glm::vec4 *pathOrigins;
__constant__ __device__ glm::vec4 *pathDirections;
__constant__ __device__ glm::vec4 *pathThroughputs;
__constant__ __device__ glm::vec3 *skybox;
__constant__ __device__ uint skyboxWidth;
__constant__ __device__ uint skyboxHeight;
__constant__ __device__ uint scrWidth;
__constant__ __device__ uint scrHeight;
__constant__ __device__ uint *blueNoise;
__constant__ __device__ float clampValue;

__constant__ __device__ PotentialContribution *connectData;
__constant__ __device__ DeviceInstanceDescriptor *instances;

#ifndef MAT_CONSTANTS_H
#define MAT_CONSTANTS_H
__constant__ __device__ DeviceMaterial *materials;
__constant__ __device__ glm::vec4 *floatTextures;
__constant__ __device__ uint *uintTextures;
#endif

#ifndef LIGHTS_H
#define LIGHTS_H
__constant__ __device__ rfw::DeviceAreaLight *areaLights;
__constant__ __device__ rfw::DevicePointLight *pointLights;
__constant__ __device__ rfw::DeviceSpotLight *spotLights;
__constant__ __device__ rfw::DeviceDirectionalLight *directionalLights;
__constant__ __device__ rfw::LightCount lightCounts;
#endif

__constant__ __device__ glm::mat4 *instance_transforms;
__constant__ __device__ glm::mat4 *inverse_transforms;

__constant__ __device__ rfw::bvh::BVHNode *topLevelBVH;
__constant__ __device__ rfw::bvh::MBVHNode *topLevelMBVH;
__constant__ __device__ uint *topPrimIndices;
__constant__ __device__ rfw::bvh::AABB *topAABBs;

__constant__ __device__ rfw::bvh::BVHNode **meshBVHs;
__constant__ __device__ rfw::bvh::MBVHNode **meshMBVHs;
__constant__ __device__ uint **meshPrimIndices;
__constant__ __device__ glm::vec4 **meshVertices;
__constant__ __device__ glm::uvec3 **meshIndices;

__host__ void setTopLevelBVH(rfw::bvh::BVHNode *ptr) { cudaMemcpyToSymbol(topLevelBVH, &ptr, sizeof(void *)); }
__host__ void setTopLevelMBVH(rfw::bvh::MBVHNode *ptr) { cudaMemcpyToSymbol(topLevelMBVH, &ptr, sizeof(void *)); }
__host__ void setTopPrimIndices(uint *ptr) { cudaMemcpyToSymbol(topPrimIndices, &ptr, sizeof(void *)); }
__host__ void setTopAABBs(rfw::bvh::AABB *ptr) { cudaMemcpyToSymbol(topAABBs, &ptr, sizeof(void *)); }

__host__ void setInstanceTransforms(glm::mat4 *ptr) { cudaMemcpyToSymbol(instance_transforms, &ptr, sizeof(void *)); }
__host__ void setInverseTransforms(glm::mat4 *ptr) { cudaMemcpyToSymbol(inverse_transforms, &ptr, sizeof(void *)); }

__host__ void setMeshBVHs(rfw::bvh::BVHNode **ptr) { cudaMemcpyToSymbol(meshBVHs, &ptr, sizeof(void *)); }
__host__ void setMeshMBVHs(rfw::bvh::MBVHNode **ptr) { cudaMemcpyToSymbol(meshMBVHs, &ptr, sizeof(void *)); }
__host__ void setMeshBVHPrimIDs(uint **ptr) { cudaMemcpyToSymbol(meshPrimIndices, &ptr, sizeof(void *)); }
__host__ void setMeshVertices(glm::vec4 **ptr) { cudaMemcpyToSymbol(meshVertices, &ptr, sizeof(void *)); }
__host__ void setMeshIndices(glm::uvec3 **ptr) { cudaMemcpyToSymbol(meshIndices, &ptr, sizeof(void *)); }

__host__ void setCameraView(rfw::CameraView *ptr) { cudaMemcpyToSymbol(view, &ptr, sizeof(void *)); }
__host__ void setCounters(Counters *ptr) { cudaMemcpyToSymbol(counters, &ptr, sizeof(void *)); }
__host__ void setAccumulator(glm::vec4 *ptr) { cudaMemcpyToSymbol(accumulator, &ptr, sizeof(void *)); }
__host__ void setStride(uint s) { cudaMemcpyToSymbol(stride, &s, sizeof(void *)); }
__host__ void setPathStates(glm::vec4 *ptr) { cudaMemcpyToSymbol(pathStates, &ptr, sizeof(void *)); }
__host__ void setPathOrigins(glm::vec4 *ptr) { cudaMemcpyToSymbol(pathOrigins, &ptr, sizeof(void *)); }
__host__ void setPathDirections(glm::vec4 *ptr) { cudaMemcpyToSymbol(pathDirections, &ptr, sizeof(void *)); }
__host__ void setPathThroughputs(glm::vec4 *ptr) { cudaMemcpyToSymbol(pathThroughputs, &ptr, sizeof(void *)); }
__host__ void setPotentialContributions(PotentialContribution *ptr) { cudaMemcpyToSymbol(connectData, &ptr, sizeof(void *)); }
__host__ void setMaterials(DeviceMaterial *ptr) { cudaMemcpyToSymbol(materials, &ptr, sizeof(void *)); }
__host__ void setFloatTextures(glm::vec4 *ptr) { cudaMemcpyToSymbol(floatTextures, &ptr, sizeof(void *)); }
__host__ void setUintTextures(uint *ptr) { cudaMemcpyToSymbol(uintTextures, &ptr, sizeof(void *)); }
__host__ void setSkybox(glm::vec3 *ptr) { cudaMemcpyToSymbol(skybox, &ptr, sizeof(void *)); }
__host__ void setSkyDimensions(uint width, uint height)
{
	cudaMemcpyToSymbol(skyboxWidth, &width, sizeof(uint));
	cudaMemcpyToSymbol(skyboxHeight, &height, sizeof(uint));
}
__host__ void setInstanceDescriptors(DeviceInstanceDescriptor *ptr) { cudaMemcpyToSymbol(instances, &ptr, sizeof(void *)); }
__host__ void setGeometryEpsilon(float value) { cudaMemcpyToSymbol(geometryEpsilon, &value, sizeof(float)); }
__host__ void setBlueNoiseBuffer(uint *ptr) { cudaMemcpyToSymbol(blueNoise, &ptr, sizeof(void *)); }
__host__ void setScreenDimensions(uint width, uint height)
{
	cudaMemcpyToSymbol(scrWidth, &width, sizeof(uint));
	cudaMemcpyToSymbol(scrHeight, &height, sizeof(uint));
}
__host__ void setLightCount(rfw::LightCount lightCount) { cudaMemcpyToSymbol(lightCounts, &lightCount, sizeof(rfw::LightCount)); }

__host__ void setAreaLights(rfw::DeviceAreaLight *als) { cudaMemcpyToSymbol(areaLights, &als, sizeof(void *)); }
__host__ void setPointLights(rfw::DevicePointLight *pls) { cudaMemcpyToSymbol(pointLights, &pls, sizeof(void *)); }
__host__ void setSpotLights(rfw::DeviceSpotLight *sls) { cudaMemcpyToSymbol(spotLights, &sls, sizeof(void *)); }
__host__ void setDirectionalLights(rfw::DeviceDirectionalLight *dls) { cudaMemcpyToSymbol(directionalLights, &dls, sizeof(void *)); }
__host__ void setClampValue(float value) { cudaMemcpyToSymbol(clampValue, &value, sizeof(float)); }

__host__ const surfaceReference *getOutputSurfaceReference()
{
	const surfaceReference *ref;
	cudaGetSurfaceReference(&ref, &output);
	return ref;
}

__global__ void initCountersExtent(uint pathCount, uint sampleIndex)
{
	if (threadIdx.x != 0)
		return; // Only run a single thread
	counters->activePaths = pathCount;
	counters->shaded = 0;		 // Thread atomic for shade kernel
	counters->extensionRays = 0; // Compaction counter for extension rays
	counters->shadowRays = 0;	 // Compaction counter for connections
	counters->totalExtensionRays = pathCount;
	counters->totalShadowRays = 0;
	counters->sampleIndex = sampleIndex;
}

__global__ void initCountersSubsequent()
{
	if (threadIdx.x != 0)
		return;
	counters->totalExtensionRays += counters->extensionRays;
	counters->activePaths = counters->extensionRays; // Remaining active paths
	counters->shaded = 0;							 // Thread atomic for shade kernel
	counters->extensionRays = 0;					 // Compaction counter for extension rays
	counters->shadowRays = 0;
}

__host__ void InitCountersForExtend(unsigned int pathCount, uint sampleIndex) { initCountersExtent<<<1, 32>>>(pathCount, sampleIndex); }
__host__ void InitCountersSubsequent() { initCountersSubsequent<<<1, 32>>>(); }

__global__ void blit_buffer(const uint scrwidth, const uint scrheight, const float scale)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= scrwidth || y >= scrheight)
		return;

	const auto index = x + y * scrwidth;
	const glm::vec4 value = accumulator[index] * scale;
	surf2Dwrite<glm::vec4>(value, output, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

__host__ cudaError blitBuffer(const unsigned int scrwidth, const unsigned int scrheight, const uint sampleID)
{
	const unsigned int alignedWidth = NEXTMULTIPLEOF(scrwidth, 16) / 16;
	const unsigned int alignedHeight = NEXTMULTIPLEOF(scrheight, 16) / 16;
	const dim3 gridDim = dim3(alignedWidth, alignedHeight, 1);
	const dim3 blockDim = dim3(16, 16, 1);

	blit_buffer<<<gridDim, blockDim>>>(scrwidth, scrheight, 1.0f / float(sampleID));
	return cudaGetLastError();
}

__device__ inline float blueNoiseSampler(int x, int y, int sampleIdx, int sampleDimension)
{
	// wrap arguments
	x &= 127;
	y &= 127;
	sampleIdx &= 255;
	sampleDimension &= 255;

	// xor index based on optimized ranking
	const int rankedSampleIndex = sampleIdx ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3];

	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];

	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];

	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

__device__ bool intersect_scene(const vec3 origin, const vec3 direction, int *instID, int *primID, float *t, vec2 *barycentrics, float t_min = 1e-5f)
{
#if !USE_TOP_MBVH
	return intersect_bvh(origin, direction, t_min, t, instID, topLevelBVH, topPrimIndices, [&](uint instance) {
		const vec3 new_origin = inverse_transforms[instance] * vec4(origin, 1);
		const vec3 new_direction = inverse_transforms[instance] * vec4(direction, 0);
		const uvec3 *indices = meshIndices[instance];
		const vec4 *vertices = meshVertices[instance];
		const uint *primIndices = meshPrimIndices[instance];

		if (indices != nullptr) // Mesh with indices
		{
#if !USE_MBVH
			return intersect_bvh(new_origin, new_direction, t_min, t, primID, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
			return intersect_mbvh(new_origin, new_direction, t_min, t, primID, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
				const uvec3 idx = indices[triangleID];
				if (intersect_triangle(new_origin, new_direction, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
				{
					const vec3 N = normalize(cross(vec3(vertices[idx.y]) - vec3(vertices[idx.x]), vec3(vertices[idx.z]) - vec3(vertices[idx.x])));
					(*barycentrics) = get_barycentrics(new_origin + (*t) * new_direction, N, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
					return true;
				}
				return false;
			});
		}

		// Intersect mesh without indices
#if !USE_MBVH
		return intersect_bvh(new_origin, new_direction, t_min, t, primID, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
		return intersect_mbvh(new_origin, new_direction, t_min, t, primID, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
			const uvec3 idx = uvec3(triangleID * 3) + uvec3(0, 1, 2);
			if (intersect_triangle(new_origin, new_direction, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
			{
				const vec3 N = normalize(cross(vec3(vertices[idx.y]) - vec3(vertices[idx.x]), vec3(vertices[idx.z]) - vec3(vertices[idx.x])));
				(*barycentrics) = get_barycentrics(new_origin + (*t) * new_direction, N, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
				return true;
			}
			return false;
		});
	});
#else
	return intersect_mbvh(origin, direction, t_min, t, instID, topLevelMBVH, topPrimIndices, [&](uint instance) {
		const vec3 new_origin = inverse_transforms[instance] * vec4(origin, 1);
		const vec3 new_direction = inverse_transforms[instance] * vec4(direction, 0);
		const uvec3 *indices = meshIndices[instance];
		const vec4 *vertices = meshVertices[instance];
		const uint *primIndices = meshPrimIndices[instance];

		if (indices != nullptr) // Mesh with indices
		{
#if !USE_MBVH
			return intersect_bvh(new_origin, new_direction, t_min, t, primID, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
			return intersect_mbvh(new_origin, new_direction, t_min, t, primID, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
				const uvec3 idx = indices[triangleID];
				if (intersect_triangle(new_origin, new_direction, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
				{
					const vec3 N = normalize(cross(vec3(vertices[idx.y]) - vec3(vertices[idx.x]), vec3(vertices[idx.z]) - vec3(vertices[idx.x])));
					(*barycentrics) = get_barycentrics(new_origin + (*t) * new_direction, N, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
					return true;
				}
				return false;
			});
		}

		// Intersect mesh without indices
#if !USE_MBVH
		return intersect_bvh(new_origin, new_direction, t_min, t, primID, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
		return intersect_mbvh(new_origin, new_direction, t_min, t, primID, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
			const uvec3 idx = uvec3(triangleID * 3) + uvec3(0, 1, 2);
			if (intersect_triangle(new_origin, new_direction, t_min, t, vertices[idx.x], vertices[idx.y], vertices[idx.z]))
			{
				const vec3 N = normalize(cross(vec3(vertices[idx.y]) - vec3(vertices[idx.x]), vec3(vertices[idx.z]) - vec3(vertices[idx.x])));
				(*barycentrics) = get_barycentrics(new_origin + (*t) * new_direction, N, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
				return true;
			}
			return false;
		});
	});
#endif
}

__device__ bool is_occluded(const vec3 origin, const vec3 direction, float t_min, float t_max)
{
#if !USE_TOP_MBVH
	return intersect_bvh_shadow(origin, direction, t_min, t_max, topLevelBVH, topPrimIndices, [&](uint instance) {
		const vec3 new_origin = inverse_transforms[instance] * vec4(origin, 1);
		const vec3 new_direction = inverse_transforms[instance] * vec4(direction, 0);
		const uvec3 *indices = meshIndices[instance];
		const vec4 *vertices = meshVertices[instance];
		const uint *primIndices = meshPrimIndices[instance];

		if (indices != nullptr) // Mesh with indices
		{
#if !USE_MBVH
			return intersect_bvh_shadow(new_origin, new_direction, t_min, t_max, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
			return intersect_mbvh_shadow(new_origin, new_direction, t_min, t_max, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
				const uvec3 idx = indices[triangleID];
				return intersect_triangle(new_origin, new_direction, t_min, &t_max, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
			});
		}

		// Intersect mesh without indices
#if !USE_MBVH
		return intersect_bvh_shadow(new_origin, new_direction, t_min, t_max, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
		return intersect_mbvh_shadow(new_origin, new_direction, t_min, t_max, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
			const uvec3 idx = uvec3(triangleID * 3) + uvec3(0, 1, 2);
			return intersect_triangle(new_origin, new_direction, t_min, &t_max, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		});
	});
#else
	return intersect_mbvh_shadow(origin, direction, t_min, t_max, topLevelMBVH, topPrimIndices, [&](uint instance) {
		const vec3 new_origin = inverse_transforms[instance] * vec4(origin, 1);
		const vec3 new_direction = inverse_transforms[instance] * vec4(direction, 0);
		const uvec3 *indices = meshIndices[instance];
		const vec4 *vertices = meshVertices[instance];

		const rfw::bvh::BVHNode *bvh = meshBVHs[instance];
		const uint *primIndices = meshPrimIndices[instance];

		if (indices != nullptr) // Mesh with indices
		{
#if !USE_MBVH
			return intersect_bvh_shadow(new_origin, new_direction, t_min, t_max, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
			return intersect_mbvh_shadow(new_origin, new_direction, t_min, t_max, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
				const uvec3 idx = indices[triangleID];
				return intersect_triangle(new_origin, new_direction, t_min, &t_max, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
			});
		}

		// Intersect mesh without indices
#if !USE_MBVH
		return intersect_bvh_shadow(new_origin, new_direction, t_min, t_max, meshBVHs[instance], primIndices, [&](uint triangleID) {
#else
		return intersect_mbvh_shadow(new_origin, new_direction, t_min, t_max, meshMBVHs[instance], primIndices, [&](uint triangleID) {
#endif
			const uvec3 idx = uvec3(triangleID * 3) + uvec3(0, 1, 2);
			return intersect_triangle(new_origin, new_direction, t_min, &t_max, vertices[idx.x], vertices[idx.y], vertices[idx.z]);
		});
	});
#endif
}

__device__ void generatePrimaryRay(const uint pathID)
{
	uint seed = WangHash(pathID * 16789 + counters->sampleIndex * 1791);

	const int sx = pathID % scrWidth;
	const int sy = pathID / scrWidth;

#if 1
	const float r0 = blueNoiseSampler(sx, sy, int(counters->sampleIndex), 0);
	const float r1 = blueNoiseSampler(sx, sy, int(counters->sampleIndex), 1);
	float r2 = blueNoiseSampler(sx, sy, int(counters->sampleIndex), 2);
	float r3 = blueNoiseSampler(sx, sy, int(counters->sampleIndex), 3);
#else
	const float r0 = RandomFloat(seed);
	const float r1 = RandomFloat(seed);
	float r2 = RandomFloat(seed);
	float r3 = RandomFloat(seed);
#endif
	const float blade = static_cast<int>(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	constexpr float piOver4point5 = 3.14159265359f / 4.5f;

	__sincosf(blade * piOver4point5, &x1, &y1);
	__sincosf((blade + 1.0f) * piOver4point5, &x2, &y2);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;

	// TODO: Calculate this on cpu
	const vec3 right = view->p2 - view->p1;
	const vec3 up = view->p3 - view->p1;

	const vec3 origin = view->pos + view->aperture * (right * xr + up * yr);

	const float u = (static_cast<float>(sx) + r0) * (1.0f / scrWidth);
	const float v = (static_cast<float>(sy) + r1) * (1.0f / scrHeight);
	const vec3 pointOnPixel = view->p1 + u * right + v * up;
	const vec3 direction = normalize(pointOnPixel - origin);

	pathOrigins[pathID] = vec4(origin, __uint_as_float((pathID << 8) + 1 /* 1 == specular */));
	pathDirections[pathID] = vec4(direction, 0);
}

__global__ void generate_rays(uint count)
{
	const uint pathID = threadIdx.x + blockIdx.x * blockDim.x;
	if (pathID >= count)
		return;
	generatePrimaryRay(pathID);
}

__global__ void intersect_rays(IntersectionStage stage, const uint pathLength, uint count)
{
	const uint pathID = threadIdx.x + blockIdx.x * blockDim.x;
	if (pathID >= count)
		return;

	if (stage == Primary || stage == Secondary)
	{
		const uint bufferIndex = pathLength % 2;
		const uint bufferID = pathID + bufferIndex * stride;

		float t = 1e34f;
		int instID;
		int primID;
		vec2 bary;

		const vec4 O4 = pathOrigins[bufferID];
		const vec4 D4 = pathDirections[bufferID];
		const vec3 O = O4;
		const vec3 D = D4;

		vec4 result = vec4(0, 0, __int_as_float(-1), 0);
		if (intersect_scene(O, D, &instID, &primID, &t, &bary))
		{
			const vec3 p = O + t * D;
			const DeviceInstanceDescriptor &desc = instances[instID];
			const rfw::DeviceTriangle &triangle = desc.triangles[primID];
			const vec3 N = normalize(desc.invTransform * vec3(triangle.vN0.w, triangle.vN1.w, triangle.vN2.w));
			const uint barycentrics = uint(65535.0f * bary.x) + (uint(65535.0f * bary.y) << 16);
			result = vec4(__uint_as_float(barycentrics), __int_as_float(uint(instID)), __int_as_float(primID), t);
		}

		pathStates[bufferID] = result;
	}
	else if (stage == Shadow)
	{
		const vec4 O4 = connectData[pathID].Origin;
		const vec4 D4 = connectData[pathID].Direction;

		const vec3 O = vec3(O4);
		const vec3 D = vec3(D4);

		if (is_occluded(O, D, 1e-5f, D4.w - 2.0f * geometryEpsilon))
			return;

		const vec4 contribution = connectData[pathID].Emission;
		const uint pixelID = __float_as_uint(contribution.w);
		accumulator[pixelID] += vec4(vec3(contribution), 1.0f);
	}
}

__global__ __launch_bounds__(128 /* Max block size */, 4 /* Min blocks per sm */) void shade_rays(const uint pathLength, uint count)
{
	const uint jobID = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobID >= count)
		return;

	const uint bufferIndex = pathLength % 2;
	const uint nextBufferIndex = 1 - bufferIndex;
	const uint bufferID = jobID + bufferIndex * stride;

	const vec4 hitData = pathStates[bufferID];
	const int primID = __float_as_int(hitData.z);

	const vec4 O4 = pathOrigins[bufferID];
	const vec4 D4 = pathDirections[bufferID];
	const uint pathID = __float_as_uint(O4.w) >> 8;
	const vec4 T4 = pathLength == 0 ? vec4(1.0f) : pathThroughputs[jobID + bufferIndex * stride];
	vec3 throughput = vec3(T4);

	if (primID < 0)
	{
		// formulas by Paul Debevec, http://www.pauldebevec.com/Probes
		const uint sky_u = static_cast<uint>(static_cast<float>(skyboxWidth - 1) * 0.5f * (1.0f + atan2(D4.x, -D4.z) * glm::one_over_pi<float>()));
		const uint sky_v = static_cast<uint>(static_cast<float>(skyboxHeight - 1) * acos(D4.y) * glm::one_over_pi<float>());
		const uint sky_idx = sky_u + sky_v * skyboxWidth;
		const vec3 skySample = skybox[sky_idx];
		vec3 contribution = throughput * vec3(skySample);

		if (any(isnan(throughput)))
			return;

		clampIntensity(contribution, clampValue);
		accumulator[pathID] += vec4(contribution, 0.0f);

		return;
	}

	const float bsdfPdf = T4.w;
	uint flags = __float_as_uint(O4.w) & 0xFF;

	const vec3 O = vec3(O4);
	const vec3 D = vec3(D4);
	const vec3 I = O + D * hitData.w;

	const uint uintBaryCentrics = __float_as_uint(hitData.x);
	const vec2 barycentrics = vec2(static_cast<float>(uintBaryCentrics & 65535), static_cast<float>(uintBaryCentrics >> 16)) * (1.0f / 65536.0f);
	const int instanceID = __float_as_int(hitData.y);
	const DeviceInstanceDescriptor &instance = instances[instanceID];
	const DeviceTriangle &triangle = instance.triangles[primID];

	vec3 N, iN, T, B;
	const ShadingData shadingData =
		getShadingData(D, barycentrics.x, barycentrics.y, view->spreadAngle * hitData.w, triangle, instanceID, N, iN, T, B, instance.invTransform);

	if (pathLength == 0 && pathID == counters->probeIdx)
	{
		counters->probedInstanceId = instanceID;
		counters->probedPrimId = primID;
		counters->probedDistance = hitData.w;
	}

	// Detect alpha in the shading code.
	if (shadingData.flags & 1)
	{
		if (pathLength < MAX_PATH_LENGTH)
		{
			if (any(isnan(throughput)))
				return;
			const uint extensionRayIdx = atomicAdd(&counters->extensionRays, 1);
			pathOrigins[extensionRayIdx + nextBufferIndex * stride] = vec4(I + D * geometryEpsilon, O4.w);
			pathDirections[extensionRayIdx + nextBufferIndex * stride] = D4;
			pathStates[extensionRayIdx + nextBufferIndex * stride] = T4;
			// TODO: this never gets hit, fix this
		}
		return;
	}

	// Terminate path on light
	if (shadingData.isEmissive()) /* r, g or b exceeds 1 */
	{
		const float DdotNL = -dot(D, N);
		vec3 contribution = vec3(0);
		if (DdotNL > 0)
		{
			if (pathLength == 0)
			{
				// Only camera rays will be treated special
				contribution = shadingData.color;
			}
			else if (flags & IS_SPECULAR)
			{
				contribution = throughput * shadingData.color;
			}
			else
			{
				// Last vertex was not specular: apply MIS
				const vec3 lastN = UnpackNormal(floatBitsToUint(D4.w));
				const float lightPdf = CalculateLightPDF(D, hitData.w, triangle.getArea(), N);
				const int triangleIdx = int(triangle.getLightTriangleIndex());
				const float pickProb = LightPickProb(triangleIdx, O, lastN, I);
				if ((bsdfPdf + lightPdf * pickProb) <= 0)
					return;

				contribution = throughput * shadingData.color * (1.0f / (bsdfPdf + lightPdf * pickProb));
			}
		}

		if (any(isnan(contribution)))
			return;

		clampIntensity(contribution, clampValue);
		accumulator[pathID] += vec4(contribution, 0.0f);
		return;
	}

	if (shadingData.getRoughness() < MIN_ROUGHNESS)
		flags |= IS_SPECULAR; // Object was specular
	else
		flags &= ~IS_SPECULAR; // Object was not specular

	uint seed = WangHash(pathID * 16789 + counters->samplesTaken * 1791 + pathLength * 720898027);
	const float flip = (dot(D, N) > 0) ? -1.0f : 1.0f;
	N *= flip;					  // Fix geometric normal
	iN *= flip;					  // Fix interpolated normal (consistent normal interpolation)
	throughput *= 1.0f / bsdfPdf; // Apply postponed bsdf pdf

	// Only cast shadow rays for non-specular objects
	if ((flags & IS_SPECULAR) == 0 &&
		(lightCounts.areaLightCount > 0 || lightCounts.pointLightCount > 0 || lightCounts.directionalLightCount > 0 || lightCounts.spotLightCount > 0))
	{
		vec3 lightColor;
		float r0, r1, pickProb, lightPdf = 0;
#if BLUENOISE
		if (counters->samplesTaken < 256)
		{
			const int x = int(pathID % scrWidth);
			const int y = int(pathID / scrWidth);
			r0 = blueNoiseSampler(blueNoise, x, y, int(counters->samplesTaken), 4);
			r1 = blueNoiseSampler(blueNoise, x, y, int(counters->samplesTaken), 5);
		}
		else
		{
			r0 = RandomFloat(seed);
			r1 = RandomFloat(seed);
		}
#else
		r0 = RandomFloat(seed);
		r1 = RandomFloat(seed);
#endif

		vec3 L = RandomPointOnLight(r0, r1, I, iN, pickProb, lightPdf, lightColor) - I;
		const float dist = length(L);
		L *= 1.0f / dist;
		const float NdotL = dot(L, iN);
		if (NdotL > 0 && lightPdf > 0)
		{
			float shadowPdf;
			const vec3 sampledBSDF = EvaluateBSDF(shadingData, iN, T, B, D * -1.0f, L, shadowPdf, seed);
			if (shadowPdf > 0)
			{
				// calculate potential contribution
				vec3 contribution = throughput * sampledBSDF * lightColor * (NdotL / (shadowPdf + lightPdf * pickProb));
				clampIntensity(contribution, clampValue);

				if (!any(isnan(contribution)))
				{
					// Add fire-and-forget shadow ray to the connections buffer
					const uint shadowRayIdx = atomicAdd(&counters->shadowRays, 1); // compaction

					connectData[shadowRayIdx].Origin = vec4(SafeOrigin(I, L, N, geometryEpsilon), 0);
					connectData[shadowRayIdx].Direction = vec4(L, dist);
					connectData[shadowRayIdx].Emission = vec4(contribution, uintBitsToFloat(pathID));
				}
			}
		}
	}

	if (pathLength >= MAX_PATH_LENGTH) // Early out in case we reached maximum path length
		return;

	vec3 R;
	float newBsdfPdf;
	vec3 bsdf = SampleBSDF(shadingData, iN, N, T, B, D * -1.0f, hitData.w, flip < 0, R, newBsdfPdf, seed);
	throughput = throughput * 1.0f / SurvivalProbability(throughput) * bsdf * abs(dot(iN, R));

#if ALLOW_DENOISER
	if (pathLength == 0)
	{
		if (counters->samplesTaken == 0)
		{
			albedos[pathIndex] = vec4(shadingData.color * abs(dot(iN, R)), 0.0f);
			normals[pathIndex] = vec4(toEyeSpace * iN, 0.0f);
		}
		else
		{
			albedos[pathIndex] += vec4(shadingData.color * abs(dot(iN, R)), 0.0f);
			normals[pathIndex] += vec4(toEyeSpace * iN, 0.0f);
		}
	}
#endif

	if (newBsdfPdf < 1e-6f || isnan(newBsdfPdf) || any(lessThan(throughput, vec3(0.0f))))
		return; // Early out in case we have an invalid bsdf

	const uint extensionRayIdx = atomicAdd(&counters->extensionRays, 1u); // Get compacted index for extension ray

	pathOrigins[extensionRayIdx + nextBufferIndex * stride] = vec4(SafeOrigin(I, R, N, geometryEpsilon), uintBitsToFloat((pathID << 8u) | flags));
	pathDirections[extensionRayIdx + nextBufferIndex * stride] = vec4(R, uintBitsToFloat(PackNormal(iN)));
	pathThroughputs[extensionRayIdx + nextBufferIndex * stride] = vec4(throughput, newBsdfPdf);
}

__host__ cudaError generateRays(const uint count)
{
	const dim3 gridDim = dim3(NEXTMULTIPLEOF(count, 64) / 64);
	const dim3 blockDim = dim3(64);

	generate_rays<<<gridDim, blockDim>>>(count);
	return cudaGetLastError();
}

__host__ cudaError intersectRays(IntersectionStage stage, const uint pathLength, const uint count)
{
	const dim3 gridDim = dim3(NEXTMULTIPLEOF(count, 64) / 64);
	const dim3 blockDim = dim3(64);

	intersect_rays<<<gridDim, blockDim>>>(stage, pathLength, count);
	return cudaGetLastError();
}

__host__ cudaError shadeRays(const uint pathLength, const uint count)
{
	const dim3 gridDim = dim3(NEXTMULTIPLEOF(count, 64) / 64);
	const dim3 blockDim = dim3(64);

	shade_rays<<<gridDim, blockDim>>>(pathLength, count);
	return cudaGetLastError();
}