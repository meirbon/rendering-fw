#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include "../../src/SharedStructs.h"
#include "getShadingData.h"
#include "random.h"
#include <Settings.h>
#include "tools.h"

#include "lights.h"

//#include "lambert.h"
#include "disney.h"
//#include "microfacet.h"

#define NEXTMULTIPLEOF(a, b) (((a) + ((b)-1)) & (0x7fffffff - ((b)-1)))
using namespace glm;

#ifndef __launch_bounds__ // Fix errors in IDE
#define __launch_bounds__(x, y)
int __float_as_int(float x) { return int(x); }
uint __float_as_uint(float x) { return uint(x); }

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
__constant__ __device__ glm::vec4 *normals;
__constant__ __device__ glm::vec4 *albedos;
__constant__ __device__ glm::vec4 *inputNormals;
__constant__ __device__ glm::vec4 *inputAlbedos;
__constant__ __device__ glm::vec4 *inputPixels;
__constant__ __device__ glm::vec4 *outputPixels;
__constant__ __device__ PotentialContribution *connectData;

__constant__ __device__ DeviceInstanceDescriptor *instances;

__global__ void initCountersExtent(unsigned int pathCount)
{
	if (threadIdx.x != 0)
		return; // Only run a single thread
	counters->activePaths = pathCount;
	counters->shaded = 0;		 // Thread atomic for shade kernel
	counters->extensionRays = 0; // Compaction counter for extension rays
	counters->shadowRays = 0;	 // Compaction counter for connections
	counters->totalExtensionRays = pathCount;
	counters->totalShadowRays = 0;
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

#define IS_SPECULAR 1

__global__ __launch_bounds__(128 /* Max block size */, 8 /* Min blocks per sm */) void shade(const uint pathLength, const glm::mat3 toEyeSpace)
{
	const int jobIndex = threadIdx.x + blockIdx.x * blockDim.x;
	if (jobIndex >= counters->activePaths)
		return;

	const uint bufferIndex = pathLength % 2;
	const uint nextBufferIndex = 1 - bufferIndex;

	const vec4 hitData = pathStates[jobIndex + bufferIndex * stride];
	const vec4 O4 = pathOrigins[jobIndex + bufferIndex * stride];
	const vec4 D4 = pathDirections[jobIndex + bufferIndex * stride];
	vec4 T4 = pathLength == 0 ? vec4(1.0f) : pathThroughputs[jobIndex + bufferIndex * stride];
	uint flags = __float_as_uint(O4.w) & 0xFF;
	vec3 throughput = vec3(T4);
	const float bsdfPdf = T4.w;

	const vec3 D = glm::vec3(D4);
	const uint pathIndex = (__float_as_uint(O4.w) >> 8u);

	const int primIdx = __float_as_int(hitData.z);
	if (primIdx < 0)
	{
		// formulas by Paul Debevec, http://www.pauldebevec.com/Probes
		const uint u = static_cast<uint>(static_cast<float>(skyboxWidth) * 0.5f * (1.0f + atan2(D.x, -D.z) * glm::one_over_pi<float>()));
		const uint v = static_cast<uint>(static_cast<float>(skyboxHeight) * acos(D.y) * glm::one_over_pi<float>());
		const uint idx = u + v * skyboxWidth;
		const vec3 skySample = idx < skyboxHeight * skyboxWidth ? skybox[idx] : vec3(0);
		vec3 contribution = throughput * 1.0f / bsdfPdf * vec3(skySample);

		if (!any(isnan(throughput)))
		{
			clampIntensity(contribution, clampValue);
			accumulator[pathIndex] += vec4(contribution, 0.0f);

			if (pathLength == 0)
			{
				if (counters->samplesTaken == 0)
				{
					albedos[pathIndex] = vec4(contribution, 0.0f);
					normals[pathIndex] = vec4(0.0f);
				}
				else
					albedos[pathIndex] += vec4(contribution, 0.0f);
			}
		}

		return;
	}

	const vec3 O = glm::vec3(O4);
	const vec3 I = O + D * hitData.w;
	const uint uintBaryCentrics = __float_as_uint(hitData.x);
	const vec2 barycentrics = vec2(static_cast<float>(uintBaryCentrics & 65535), static_cast<float>(uintBaryCentrics >> 16)) * (1.0f / 65536.0f);
	const int instanceIdx = __float_as_uint(hitData.y);
	const DeviceInstanceDescriptor &instance = instances[instanceIdx];

	const DeviceTriangle &triangle = instance.triangles[primIdx];

	glm::vec3 N, iN, T, B;
	const ShadingData shadingData =
		getShadingData(D, barycentrics.x, barycentrics.y, view->spreadAngle * hitData.w, triangle, instanceIdx, N, iN, T, B, instance.invTransform);

	if (counters->samplesTaken == 0 && pathLength == 0 && pathIndex == counters->probeIdx)
	{
		counters->probedInstanceId = instanceIdx;
		counters->probedPrimId = primIdx;
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

		if (pathLength == 0)
		{
			const vec3 albedo = min(contribution, vec3(1.0f));
			if (counters->samplesTaken == 0)
			{
				albedos[pathIndex] = vec4(albedo, 0.0f);
				normals[pathIndex] = vec4(toEyeSpace * iN, 0.0f);
			}
			else
			{
				albedos[pathIndex] += vec4(albedo, 0.0f);
				normals[pathIndex] += vec4(toEyeSpace * iN, 0.0f);
			}
		}

		clampIntensity(contribution, clampValue);
		accumulator[pathIndex] += vec4(contribution, 0.0f);
		return;
	}

	if (shadingData.getRoughness() < MIN_ROUGHNESS)
		flags |= IS_SPECULAR; // Object was specular
	else
		flags &= ~IS_SPECULAR; // Object was not specular

	uint seed = WangHash(pathIndex * 16789 + counters->samplesTaken * 1791 + pathLength * 720898027);
	const float flip = (dot(D, N) > 0) ? -1.0f : 1.0f;
	N *= flip;					  // Fix geometric normal
	iN *= flip;					  // Fix interpolated normal (consistent normal interpolation)
	throughput *= 1.0f / bsdfPdf; // Apply postponed bsdf pdf

	if ((flags & IS_SPECULAR) == 0 && (lightCounts.areaLightCount + lightCounts.pointLightCount + lightCounts.directionalLightCount +
									   lightCounts.spotLightCount) > 0) // Only cast shadow rays for non-specular objects
	{
		vec3 lightColor;
		float r0, r1, pickProb, lightPdf = 0;
#if BLUENOISE
		if (counters->samplesTaken < 256)
		{
			const int x = int(pathIndex % scrWidth);
			const int y = int(pathIndex / scrWidth);
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
			const vec3 sampledBSDF = EvaluateBSDF(shadingData, iN, T, B, D * -1.0f, L, shadowPdf);
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
					connectData[shadowRayIdx].Emission = vec4(contribution, uintBitsToFloat(pathIndex));
				}
			}
		}
	}

	if (pathLength >= MAX_PATH_LENGTH) // Early out in case we reached maximum path length
		return;

	vec3 R, bsdf;
	float newBsdfPdf = 0.0f, r3, r4;
#if BLUENOISE						  // TODO
	if (counters->samplesTaken < 256) // Blue noise
	{
		const int x = int(pathIndex % scrWidth) & 127;
		const int y = int(pathIndex / scrWidth) & 127;
		r3 = blueNoiseSampler(blueNoise, x, y, int(counters->samplesTaken), 4);
		r4 = blueNoiseSampler(blueNoise, x, y, int(counters->samplesTaken), 5);
	}
	else
	{
		r3 = RandomFloat(seed);
		r4 = RandomFloat(seed);
	}
#else
	r3 = RandomFloat(seed);
	r4 = RandomFloat(seed);
#endif

	bsdf = SampleBSDF(shadingData, iN, N, T, B, D * -1.0f, hitData.w, flip < 0, r3, r4, R, newBsdfPdf);
	throughput = throughput * 1.0f / SurvivalProbability(throughput) * bsdf * abs(dot(iN, R));

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

	if (newBsdfPdf < 1e-6f || isnan(newBsdfPdf) || any(isnan(throughput)) || all(lessThanEqual(throughput, vec3(0.0f))))
		return; // Early out in case we have an invalid bsdf

	const uint extensionRayIdx = atomicAdd(&counters->extensionRays, 1u); // Get compacted index for extension ray

	pathOrigins[extensionRayIdx + nextBufferIndex * stride] = vec4(I + R * geometryEpsilon, uintBitsToFloat((pathIndex << 8u) | flags));
	pathDirections[extensionRayIdx + nextBufferIndex * stride] = vec4(R, uintBitsToFloat(PackNormal(iN)));
	pathThroughputs[extensionRayIdx + nextBufferIndex * stride] = vec4(throughput, newBsdfPdf);
}

__global__ void finalize(const uint scrwidth, const uint scrheight, const float pixelValueScale)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= scrwidth || y >= scrheight)
		return;

	const auto index = x + y * scrwidth;

	const glm::vec3 normal = vec3(normals[index]) * pixelValueScale;
	const glm::vec3 albedo = vec3(albedos[index]) * pixelValueScale;
	inputNormals[index] = vec4(normal, 1.0f);
	inputAlbedos[index] = vec4(albedo, 1.0f);

	const glm::vec4 value = accumulator[index] * pixelValueScale;
	inputPixels[index] = value;
}

__global__ void finalizeBlit(const uint scrwidth, const uint scrheight, const float pixelValueScale)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= scrwidth || y >= scrheight)
		return;

	const auto index = x + y * scrwidth;

	const glm::vec4 value = accumulator[index] * pixelValueScale;

	surf2Dwrite<glm::vec4>(value, output, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

__global__ void tonemap(const uint scrwidth, const uint scrheight, const float pixelValueScale, const float brightness, const float contrastFactor)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;

	if (x >= scrwidth || y >= scrheight)
		return;

	const auto index = x + y * scrwidth;
	const glm::vec4 value = outputPixels[index];
	const float r = sqrt(max(0.0f, (value.x - 0.5f) * contrastFactor + 0.5f + brightness));
	const float g = sqrt(max(0.0f, (value.y - 0.5f) * contrastFactor + 0.5f + brightness));
	const float b = sqrt(max(0.0f, (value.z - 0.5f) * contrastFactor + 0.5f + brightness));
	surf2Dwrite<glm::vec4>(glm::vec4(r, g, b, value.w), output, x * sizeof(float4), y, cudaBoundaryModeClamp);
}

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
__host__ void setNormalBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(normals, &ptr, sizeof(void *)); }
__host__ void setAlbedoBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(albedos, &ptr, sizeof(void *)); }
__host__ void setInputNormalBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(inputNormals, &ptr, sizeof(void *)); }
__host__ void setInputAlbedoBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(inputAlbedos, &ptr, sizeof(void *)); }
__host__ void setInputPixelBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(inputPixels, &ptr, sizeof(void *)); }
__host__ void setOutputPixelBuffer(glm::vec4 *ptr) { cudaMemcpyToSymbol(outputPixels, &ptr, sizeof(void *)); }

__host__ const surfaceReference *getOutputSurfaceReference()
{
	const surfaceReference *ref;
	cudaGetSurfaceReference(&ref, &output);
	return ref;
}

__host__ void InitCountersForExtend(unsigned int pathCount) { initCountersExtent<<<1, 32>>>(pathCount); }
__host__ void InitCountersSubsequent() { initCountersSubsequent<<<1, 32>>>(); }

__host__ cudaError launchShade(const uint pathCount, const uint pathLength, const glm::mat3 &toEyeSpace)
{
	const dim3 gridDim = dim3(NEXTMULTIPLEOF(pathCount, 128) / 128, 1, 1);
	const dim3 blockDim = dim3(128, 1, 1);
	shade<<<gridDim, blockDim>>>(pathLength, toEyeSpace);

	return cudaGetLastError();
}

__host__ cudaError launchFinalize(bool blit, const unsigned int scrwidth, const unsigned int scrheight, const unsigned int samples, const float brightness,
								  const float contrast)
{
	const unsigned int alignedWidth = NEXTMULTIPLEOF(scrwidth, 16);
	const unsigned int alignedHeight = NEXTMULTIPLEOF(scrheight, 16);
	const dim3 gridDim = dim3(alignedWidth, alignedHeight, 1);
	const dim3 blockDim = dim3(16, 16, 1);

	if (blit)
		finalizeBlit<<<gridDim, blockDim>>>(scrwidth, scrheight, 1.0f / float(samples));
	else
		finalize<<<gridDim, blockDim>>>(scrwidth, scrheight, 1.0f / float(samples));

	return cudaGetLastError();
}

cudaError launchTonemap(unsigned int scrwidth, unsigned int scrheight, unsigned int samples, float brightness, float contrast)
{
	const unsigned int alignedWidth = NEXTMULTIPLEOF(scrwidth, 16);
	const unsigned int alignedHeight = NEXTMULTIPLEOF(scrheight, 16);
	const dim3 gridDim = dim3(alignedWidth, alignedHeight, 1);
	const dim3 blockDim = dim3(16, 16, 1);

	const float contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
	tonemap<<<gridDim, blockDim>>>(scrwidth, scrheight, 1.0f / float(samples), brightness, contrastFactor);

	return cudaGetLastError();
}