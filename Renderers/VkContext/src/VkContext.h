//
// Created by Mèir Noordermeer on 24/08/2019.
//

#ifndef RENDERING_FW_VKCONTEXT_VKCONTEXT_H
#define RENDERING_FW_VKCONTEXT_VKCONTEXT_H

#include <vulkan/vulkan.hpp>

#include <RenderContext.h>
#include <ContextExport.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>

#include "ShaderModule.h"
#include "VulkanDevice.h"
#include "SwapChain.h"
#include "FrameBuffer.h"
#include "RenderPass.h"
#include "VkMesh.h"

namespace vkc
{

class VkContext : public rfw::RenderContext
{
  public:
	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override
	{
		using namespace rfw;
		return { OPENGL_TEXTURE, WINDOW, VULKAN_TEXTURE };
	}
	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void cleanup() override;

	void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void setMaterials(const std::vector<rfw::DeviceMaterial> &materials,
					  const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void setTextures(const std::vector<rfw::TextureData> &textures) override;
	void setMesh(size_t index, const rfw::Mesh &mesh) override;
	void setInstance(size_t instanceIdx, size_t mesh, const mat4 &transform) override;
	void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
				   const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
				   const rfw::DeviceDirectionalLight *directionalLights) override;
	void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	[[nodiscard]] rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	rfw::RenderStats getStats() const override;

  private:
	std::shared_ptr<rfw::utils::Window> m_Window;

	uint32_t m_CurrentFrame = 0;
	vk::Instance m_Instance = nullptr;
	vk::DispatchLoaderDynamic m_LoaderDynamic = {};
	vk::DebugUtilsMessengerEXT m_DebugMessenger = {};

	VulkanDevice m_Device;
	SwapChain *m_SwapChain = nullptr;
	vk::SurfaceKHR m_Surface = {};
	vk::PipelineLayout m_PipelineLayout = {};
	vk::Pipeline m_GraphicsPipeline = {};

	std::vector<vkc::VkMesh*> m_Meshes;
	std::vector<vk::Buffer> m_VertexBuffers;
	std::vector<vk::Framebuffer> m_Framebuffers = {};
	std::vector<vk::CommandBuffer> m_CommandBuffers = {};
	std::vector<vk::Semaphore> m_ImageAvailableSemaphores = {};
	std::vector<vk::Semaphore> m_RenderFinishedSemaphores = {};
	RenderPass *m_RenderPass = nullptr;

	void setupValidationLayers(vk::InstanceCreateInfo *createInfo) const;
	void setupDebugMessenger();
	[[nodiscard]] vk::PhysicalDevice pickPhysicalDevice() const;

	void createSurface();
	void createRenderPass();
	void createFrameBuffers();
	void createGraphicsPipeline();
	void printAvailableExtensions() const;
	void createCommandBuffers();
	void createSemaphores();

	static std::vector<const char *> getRequiredExtensions();
};

} // namespace vkc

#endif // RENDERING_FW_VKCONTEXT_VKCONTEXT_H
