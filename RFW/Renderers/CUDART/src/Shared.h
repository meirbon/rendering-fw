#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <cuda_runtime.h>

#include <Structures.h>

#include <DeviceStructures.h>

#include "bvh/BVHNode.h"
#include "bvh/MBVHNode.h"

using namespace glm;
using namespace rfw;

struct InstanceBVHDescriptor
{
	const bvh::MBVHNode *mbvh;		 // 8
	const uint *bvh_indices;		 // 16
	const vec4 *vertices;			 // 24
	const uvec3 *indices;			 // 32
	const DeviceTriangle *triangles; // 36
	mat4 instance_transform;   // 100
	mat3x4 inverse_transform;  // 148
	const bvh::BVHNode *bvh;		 // 156

	// glm::mat4 inverse_transform;  //
};

struct PotentialContribution
{
	glm::vec4 Origin;
	glm::vec4 Direction;
	glm::vec4 Emission;
};

struct Counters
{
	unsigned int activePaths;
	unsigned int shaded;
	unsigned int extensionRays;
	unsigned int shadowRays;

	unsigned int totalExtensionRays;
	unsigned int totalShadowRays;
	unsigned int samplesTaken;
	unsigned int probeIdx;

	int probedInstanceId;
	int probedPrimId;
	float probedDistance;
	float dummy0;

	glm::vec3 probedPoint;
	unsigned int sampleIndex;
};

enum IntersectionStage
{
	Primary,
	Secondary,
	Shadow
};

void setTopLevelBVH(rfw::bvh::BVHNode *ptr);
void setTopLevelMBVH(rfw::bvh::MBVHNode *ptr);
void setTopPrimIndices(uint *ptr);
void setTopAABBs(rfw::bvh::AABB *ptr);
// void setInstanceTransforms(glm::mat4 *ptr);
// void setInverseTransforms(glm::mat4 *ptr);

void setInstances(InstanceBVHDescriptor *ptr);

// void setMeshBVHs(rfw::bvh::BVHNode **ptr);
// void setMeshMBVHs(rfw::bvh::MBVHNode **ptr);
// void setMeshBVHPrimIDs(uint **ptr);
// void setMeshVertices(glm::vec4 **ptr);
// void setMeshIndices(glm::uvec3 **ptr);

void setCameraView(rfw::CameraView *ptr);
void setCounters(Counters *ptr);
void setAccumulator(glm::vec4 *ptr);
void setStride(uint s);
void setPathStates(glm::vec4 *ptr);
void setPathOrigins(glm::vec4 *ptr);
void setPathDirections(glm::vec4 *ptr);
void setPathThroughputs(glm::vec4 *ptr);
void setPotentialContributions(PotentialContribution *ptr);

void setMaterials(DeviceMaterial *ptr);
void setFloatTextures(glm::vec4 *ptr);
void setUintTextures(uint *ptr);
void setSkybox(glm::vec3 *ptr);
void setSkyDimensions(uint width, uint height);
void setInstanceDescriptors(DeviceInstanceDescriptor *ptr);
void setGeometryEpsilon(float value);
void setBlueNoiseBuffer(uint *ptr);
void setScreenDimensions(uint width, uint height);
void setLightCount(rfw::LightCount lightCount);
void setAreaLights(rfw::DeviceAreaLight *als);
void setPointLights(rfw::DevicePointLight *pls);
void setSpotLights(rfw::DeviceSpotLight *sls);
void setDirectionalLights(rfw::DeviceDirectionalLight *dls);
void setClampValue(float value);
const surfaceReference *getOutputSurfaceReference();
void InitCountersForExtend(uint pathCount, uint sampleIndex);
void InitCountersSubsequent();

// Kernels
cudaError blitBuffer(uint scrwidth, uint scrheight, const uint sampleID);
cudaError intersectRays(IntersectionStage stage, const uint pathLength, const uint count);
cudaError shadeRays(const uint pathLength, const uint count);