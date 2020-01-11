#pragma once

#include <vulkan/vulkan.hpp>
#include <MathIncludes.h>
#include <vk_mem_alloc.h>

#include <array>
#include <memory>
#include <utility>
#include <vector>
#include <set>
#include <map>
#include <string>

#include <GL/glew.h>

#include <utils.h>
#include <BlueNoise.h>
#include <utils/gl/CheckGL.h>
#include <utils/Timer.h>
#include <utils/File.h>
#include <utils/Logger.h>

#include <utils/LibExport.h>
#include <RenderContext.h>
#include <ContextExport.h>

#include <vulkan/vulkan.hpp>
#include <MathIncludes.h>

#include <Structures.h>
#include <DeviceStructures.h>

#include "CheckVK.h"

#include "VulkanDevice.h"
#include "Buffer.h"
#include "VmaBuffer.h"
#include "Mesh.h"
#include "ComputePipeline.h"
#include "DescriptorSet.h"
#include "InteropTexture.h"
#include "UniformObject.h"
#include "RtxPipeline.h"
#include "Image.h"
#include "AccelerationStructure.h"
#include "Shader.h"

namespace vkrtx
{

/**
 * Uniform objects need to be at least 8-byte aligned thus using float3 instead of float4 is not an option.
 */
struct VkCamera
{
	VkCamera() = default;
	VkCamera(const rfw::CameraView &view, uint samplesTaken, float epsilon, uint width, uint height);

	vec4 pos_lensSize;
	vec4 right_spreadAngle;
	vec4 up;
	vec4 p1;
	uint samplesTaken;
	float geometryEpsilon;
	int scrwidth;
	int scrheight;
};

struct FinalizeParams
{
	FinalizeParams() = default;
	FinalizeParams(int w, int h, int samplespp, float brightness, float contrast);

	uint scrwidth = 0;
	uint scrheight = 0;
	uint spp = 0;
	uint idummy = 0;
	float pixelValueScale = 0;
	float brightness = 0;
	float contrastFactor = 0;
	float dummy = 0;
};

struct PotentialContribution
{
	glm::vec4 Origin;
	glm::vec4 Direction;
	glm::vec4 Emission;
};

// counters and other global data, in device memory
struct Counters // 14 counters
{
	void Reset(rfw::LightCount lightCount, uint scrwidth, uint scrheight, float ClampValue = 10.0f)
	{
		pathLength = 0;
		scrWidth = scrwidth;
		scrHeight = scrheight;
		pathCount = scrwidth * scrheight;
		generated = 0;
		extensionRays = 0;
		shadowRays = 0;
		probePixelIdx = 0;
		probedInstid = 0;
		probedTriid = 0;
		probedDist = 0;
		clampValue = ClampValue;
		lightCounts = uvec4(lightCount.areaLightCount, lightCount.pointLightCount, lightCount.spotLightCount, lightCount.directionalLightCount);
	}

	uint pathLength;
	uint scrWidth;
	uint scrHeight;
	uint pathCount;

	uint generated;
	uint extensionRays;
	uint shadowRays;
	uint probePixelIdx;

	int probedInstid;
	int probedTriid;
	float probedDist;
	float clampValue;

	glm::uvec4 lightCounts;
};

class Context : public rfw::RenderContext
{
  public:
	Context();
	~Context() override;

	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override
	{
		using namespace rfw;
		return {OPENGL_TEXTURE, WINDOW};
	};

	// Initialization methods, by default these throw to indicate the chosen rendercontext does not support the
	// specified target
	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void setTextures(const std::vector<rfw::TextureData> &textures) override;
	void setMesh(size_t index, const rfw::Mesh &mesh) override;
	void setInstance(size_t index, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
				   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights) override;
	void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	[[nodiscard]] rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	rfw::RenderStats getStats() const override;

  private:
	// internal methods
	void initRenderer();
	void createInstance();
	static void setupValidationLayers(vk::InstanceCreateInfo &createInfo);

	void createDebugReportCallback();
	void createDevice();
	void createCommandBuffers();
	void resizeBuffers();

	void createRayTracingPipeline();
	void createShadePipeline();
	void createFinalizePipeline();
	void createDescriptorSets();
	void recordCommandBuffers();
	void createBuffers();
	void initializeDescriptorSets();

	vk::Instance m_VkInstance = nullptr;
	vk::DebugUtilsMessengerEXT m_VkDebugMessenger = nullptr; // Debug validation messenger
	vk::CommandBuffer m_BlitCommandBuffer;
	std::vector<GeometryInstance> m_Instances;
	std::vector<bool> m_MeshChanged = std::vector<bool>(256);
	std::vector<vk::DescriptorBufferInfo> m_TriangleBufferInfos;
	std::vector<vkrtx::Mesh *> m_Meshes{};
	VulkanDevice m_Device;

	// Frame data
	InteropTexture *m_InteropTexture = nullptr;
	uint32_t m_SamplesPP{};
	uint32_t m_CurrentFrame = 0;
	uint32_t m_ScrWidth = 0, m_ScrHeight = 0;
	int m_Initialized = false;
	int m_First = true;
	int m_InstanceMeshMappingDirty = true;
	int m_SamplesTaken = 0;
	rfw::LightCount m_LightCounts{};
	bool m_FirstConvergingFrame = false;

	// Uniform data
	UniformObject<VkCamera> *m_UniformCamera{};
	UniformObject<FinalizeParams> *m_UniformFinalizeParams{};
	TopLevelAS *m_TopLevelAS = nullptr;

	// Ray trace pipeline
	DescriptorSet *rtDescriptorSet = nullptr;
	RTXPipeline *rtPipeline = nullptr;

	// Shade pipeline
	DescriptorSet *shadeDescriptorSet = nullptr;
	ComputePipeline *shadePipeline = nullptr;

	// finalize pipeline
	DescriptorSet *finalizeDescriptorSet = nullptr;
	ComputePipeline *finalizePipeline = nullptr;

	// Storage buffers
	VmaBuffer<mat4> *m_InvTransformsBuffer = nullptr;
	std::vector<mat4> m_InvTransforms;

	std::vector<rfw::TextureData> m_TexDescriptors;
	std::vector<size_t> m_InstanceMeshIndices;
	VmaBuffer<uint> *m_RGBA32Buffer = nullptr;
	VmaBuffer<glm::vec4> *m_RGBA128Buffer = nullptr;
	Counters m_HostCounters;
	VmaBuffer<uint8_t> *m_ScratchBuffer;
	VmaBuffer<Counters> *m_Counters = nullptr;
	VmaBuffer<rfw::DeviceMaterial> *m_Materials = nullptr;
	VmaBuffer<rfw::DeviceAreaLight> *m_AreaLightBuffer = nullptr;
	VmaBuffer<rfw::DevicePointLight> *m_PointLightBuffer = nullptr;
	VmaBuffer<rfw::DeviceSpotLight> *m_SpotLightBuffer = nullptr;
	VmaBuffer<rfw::DeviceDirectionalLight> *m_DirectionalLightBuffer = nullptr;
	VmaBuffer<uint> *m_BlueNoiseBuffer = nullptr;
	VmaBuffer<glm::vec4> *m_CombinedStateBuffer[2] = {nullptr, nullptr};
	VmaBuffer<glm::vec4> *m_AccumulationBuffer = nullptr;
	VmaBuffer<PotentialContribution> *m_PotentialContributionBuffer = nullptr;

	Image *m_SkyboxImage = nullptr;
	Image *m_OffscreenImage = nullptr; // Off-screen render image
	glm::uvec2 m_ProbePos = glm::uvec2(0);
	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedTriangle = 0;
	float m_ProbedDist = -1.0f;
	float m_GeometryEpsilon = 1e-4f;

	rfw::RenderStats m_Stats;
};

} // namespace vkrtx
