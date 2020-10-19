#pragma once

#include <cuda_gl_interop.h>

#include <DeviceStructures.h>

void setCameraView(rfw::CameraView *ptr);
void setCounters(Counters *ptr);
void setAccumulator(glm::vec4 *ptr);
void setStride(uint s);
void setPathStates(glm::vec4 *ptr);
void setPathOrigins(glm::vec4 *ptr);
void setPathDirections(glm::vec4 *ptr);
void setPathThroughputs(glm::vec4 *ptr);
void setPotentialContributions(PotentialContribution *ptr);
void setMaterials(rfw::DeviceMaterial *ptr);
void setFloatTextures(glm::vec4 *ptr);
void setUintTextures(uint *ptr);
void setSkybox(glm::vec3 *ptr);
void setSkyDimensions(uint width, uint height);
void setInstanceDescriptors(rfw::DeviceInstanceDescriptor *ptr);
void setGeometryEpsilon(float value);
void setBlueNoiseBuffer(uint *ptr);
void setScreenDimensions(uint width, uint height);
void setLightCount(rfw::LightCount lightCount);
void setAreaLights(rfw::DeviceAreaLight *areaLights);
void setPointLights(rfw::DevicePointLight *pointLights);
void setSpotLights(rfw::DeviceSpotLight *spotLights);
void setDirectionalLights(rfw::DeviceDirectionalLight *directionalLights);
void setClampValue(float value);
void setNormalBuffer(glm::vec4 *ptr);
void setAlbedoBuffer(glm::vec4 *ptr);
void setInputNormalBuffer(glm::vec4 *ptr);
void setInputAlbedoBuffer(glm::vec4 *ptr);
void setInputPixelBuffer(glm::vec4 *ptr);
void setOutputPixelBuffer(glm::vec4 *ptr);

const surfaceReference *getOutputSurfaceReference();
void InitCountersForExtend(unsigned int pathCount);
void InitCountersSubsequent();

cudaError launchShade(uint pathCount, uint pathLength, const glm::mat3 &toEyeSpace);
cudaError launchFinalize(bool blit, unsigned int scrwidth, unsigned int scrheight, unsigned int samples, float brightness, float contrast);
cudaError launchTonemap(unsigned int scrwidth, unsigned int scrheight, unsigned int samples, float brightness, float contrast);
cudaError blitBuffer(const unsigned int scrwidth, const unsigned int scrheight);