//
// Created by Mèir Noordermeer on 24/08/2019.
//

#define GLFW_INCLUDE_VULKAN
#include "VkContext.h"

#include <utils/Logger.h>
#include <utils/File.h>
#include <GLFW/glfw3.h>

#include <vector>
#include <cstring>
#include <map>

#include "CheckVK.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"

const std::vector<const char *> validationLayers = {"VK_LAYER_LUNARG_standard_validation"
#ifdef __linux__
													,
													"VK_LAYER_PRIMUS_PrimusVK"
#endif
};

#ifndef NDEBUG
constexpr bool IS_DEBUG = true;
#else
constexpr bool IS_DEBUG = false;
#endif

using namespace rfw;
using namespace vkc;

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
													const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
	std::string severity;
	std::string type;

	switch (messageSeverity)
	{
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT):
		severity = "VERBOSE";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT):
		severity = "INFO";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT):
		severity = "WARNING";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT):
		severity = "ERROR";
		break;
	default:
		break;
	}

	switch (messageType)
	{
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT):
		type = "GENERAL";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT):
		type = "VALIDATION";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT):
		type = "PERFORMANCE";
		break;
	default:
		break;
	}

	if (severity == "INFO")
	{
		utils::logger::log("Validation Layer (%s, %s) : \"%s\"\n", severity.c_str(), type.c_str(), pCallbackData->pMessage);
	}
	else
	{
		utils::logger::warning("Validation Layer (%s, %s) : \"%s\"\n", severity.c_str(), type.c_str(), pCallbackData->pMessage);
	}

	return VK_FALSE;
}

void VkContext::init(std::shared_ptr<utils::Window> &window)
{
	m_Window = window;
	if (!glfwVulkanSupported())
		throw std::runtime_error("GLFW instance does not support Vulkan.");
	vk::ApplicationInfo appInfo =
		vk::ApplicationInfo("Rendering FW", VK_MAKE_VERSION(1u, 0u, 0u), "No Engine", VK_MAKE_VERSION(1u, 0u, 0u), VK_API_VERSION_1_1);

	vk::InstanceCreateInfo createInfo{};
	createInfo.setPApplicationInfo(&appInfo);

	const auto extensions = VkContext::getRequiredExtensions();
	createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()));
	createInfo.setPpEnabledExtensionNames(extensions.data());

	setupValidationLayers(&createInfo);

	const auto result = vk::createInstance(&createInfo, nullptr, &m_Instance);

	if (result == vk::Result::eSuccess)
		utils::logger::log("VkContext", "Successfully initialized Vulkan.");
	else
		throw std::runtime_error("Could not create Vulkan instance.");

	printAvailableExtensions();
	createSurface();
	const auto device = pickPhysicalDevice();

	m_Device = VulkanDevice(m_Instance, device, {VK_KHR_SWAPCHAIN_EXTENSION_NAME}, m_Surface);
	m_LoaderDynamic = vk::DispatchLoaderDynamic(m_Instance, m_Device.getVkDevice());
	setupDebugMessenger();

	m_SwapChain = new SwapChain(m_Device, m_Surface);
	m_SwapChain->create(m_Window->getWidth(), m_Window->getHeight());

	createRenderPass();
	createFrameBuffers();
	createGraphicsPipeline();

	createCommandBuffers();
	createSemaphores();
}

void VkContext::cleanup()
{
	if (m_DebugMessenger)
	{
		m_Instance.destroyDebugUtilsMessengerEXT(m_DebugMessenger, nullptr, m_LoaderDynamic);
		m_DebugMessenger = nullptr;
	}

	if (m_GraphicsPipeline)
	{
		m_Device->destroyPipeline(m_GraphicsPipeline);
		m_GraphicsPipeline = nullptr;
	}
	if (m_PipelineLayout)
	{
		m_Device->destroyPipelineLayout(m_PipelineLayout);
		m_PipelineLayout = nullptr;
	}

	for (auto framebuffer : m_Framebuffers)
		m_Device->destroyFramebuffer(framebuffer);
	m_Framebuffers.clear();

	if (m_RenderPass)
		delete m_RenderPass, m_RenderPass = nullptr;
	if (m_SwapChain)
		delete m_SwapChain, m_SwapChain = nullptr;

	for (auto m_ImageAvailableSemaphore : m_ImageAvailableSemaphores)
		m_Device->destroySemaphore(m_ImageAvailableSemaphore);
	for (auto m_RenderFinishedSemaphore : m_RenderFinishedSemaphores)
		m_Device->destroySemaphore(m_RenderFinishedSemaphore);

	m_ImageAvailableSemaphores.clear();
	m_RenderFinishedSemaphores.clear();

	if (!m_CommandBuffers.empty())
		m_Device.freeCommandBuffers(m_CommandBuffers, VulkanDevice::GRAPHICS);
	m_CommandBuffers.clear();

	for (auto *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();

	m_Device.cleanup();
	if (m_Surface)
	{
		m_Instance.destroySurfaceKHR(m_Surface);
		m_Surface = nullptr;
	}
}

void VkContext::renderFrame(const rfw::Camera &camera, RenderStatus status)
{
	uint32_t imageIndex;
	vk::Result error = m_SwapChain->acquireNextImage(m_ImageAvailableSemaphores.at(m_CurrentFrame), &imageIndex);
	if (error == vk::Result::eErrorOutOfDateKHR)
	{
		m_SwapChain->create(m_Window->getWidth(), m_Window->getHeight());
		createCommandBuffers();
		return;
	}
	else
		CheckVK(error);

	const vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	vk::SubmitInfo submitInfo = vk::SubmitInfo(1, &m_ImageAvailableSemaphores.at(m_CurrentFrame), &waitStage, 1, &m_CommandBuffers.at(m_CurrentFrame), 1,
											   &m_RenderFinishedSemaphores.at(m_CurrentFrame));

	const auto graphicsQueue = m_Device.getGraphicsQueue();

	// Submit render command to graphics queue
	graphicsQueue.submit(1, &submitInfo, nullptr);

	graphicsQueue.waitIdle();

	// Queue present but wait till render is finished
	error = m_SwapChain->queuePresent(m_Device.getPresentQueue(), imageIndex, m_RenderFinishedSemaphores.at(m_CurrentFrame));
	if (error == vk::Result::eErrorOutOfDateKHR)
	{
		m_SwapChain->create(m_Window->getWidth(), m_Window->getHeight());
		createCommandBuffers();
		return;
	}
	else
		CheckVK(error);

	m_CurrentFrame = (m_CurrentFrame + 1u) % static_cast<uint>(m_SwapChain->getBuffers().size());
}

std::vector<const char *> VkContext::getRequiredExtensions()
{
	uint32_t extensionCount = 0;
	const char **glfwExtensions = glfwGetRequiredInstanceExtensions(&extensionCount);
	std::vector<const char *> extensions = std::vector<const char *>(glfwExtensions, glfwExtensions + extensionCount);

	if constexpr (IS_DEBUG)
		extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

	return extensions;
}

void VkContext::setupValidationLayers(vk::InstanceCreateInfo *createInfo) const
{
	// Get supported layers
	const std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();

	const auto hasLayer = [&availableLayers](const char *layerName) -> bool {
		for (const auto &layer : availableLayers)
			if (strcmp(layerName, layer.layerName) == 0)
				return true;

		return false;
	};

	if (!availableLayers.empty())
	{
		utils::logger::log("VkContext", "Available layers:");
		for (const auto &layer : availableLayers)
			std::cout << "\t" << layer.layerName << std::endl << "\t\tdescription: " << layer.description << std::endl;
	}

	if (IS_DEBUG)
	{
		bool layersFound = true;
		for (auto layer : validationLayers)
		{
			if (!hasLayer(layer))
				WARNING("VkContext: Could not enable validation layer: \"%s\"", layer), layersFound = false;
		}

		if (layersFound)
		{
			// All layers available
			createInfo->setEnabledLayerCount(static_cast<uint32_t>(validationLayers.size()));
			createInfo->setPpEnabledLayerNames(validationLayers.data());

			std::string layers = validationLayers.at(0);
			for (size_t i = 1; i < validationLayers.size(); i++)
			{
				layers += ", ";
				layers += validationLayers.at(i);
			}

			DEBUG("Enabled validation layers: %s", layers.c_str());
		}
	}
	else
	{
		createInfo->setEnabledLayerCount(0);
	}
}

void VkContext::setupDebugMessenger()
{
	if (!IS_DEBUG)
		return;

	vk::DebugUtilsMessengerCreateInfoEXT createInfo{};
	createInfo.messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
								 vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
	createInfo.messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;
	createInfo.pfnUserCallback = debugCallback;
	createInfo.pUserData = nullptr;

	m_Instance.createDebugUtilsMessengerEXT(createInfo, nullptr, m_LoaderDynamic);
	DEBUG("Created debug utils messenger.");
}

vk::PhysicalDevice VkContext::pickPhysicalDevice() const
{
	vk::PhysicalDevice device;

	const auto physicalDevices = m_Instance.enumeratePhysicalDevices();
	if (physicalDevices.empty())
		throw std::runtime_error("No Vulkan devices available.");

	std::multimap<unsigned int, vk::PhysicalDevice> candidates;
	for (const auto &dev : physicalDevices)
		candidates.insert(std::make_pair(VulkanDevice::rateDevice(dev, m_Surface), dev));

	if (candidates.rbegin()->first > 0)
		device = candidates.rbegin()->second;
	else
		throw std::runtime_error("No suitable Vulkan device available.");

	assert(device);
	return device;
}

void VkContext::createSurface()
{
	vk::SurfaceKHR surface;
	if (glfwCreateWindowSurface((VkInstance)m_Instance, m_Window->getGLFW(), nullptr, reinterpret_cast<VkSurfaceKHR *>(&surface)) != VK_SUCCESS)
		throw std::runtime_error("Failed to create window surface.");

	m_Surface = surface;

	assert(m_Surface);
}

void VkContext::createRenderPass()
{
	m_RenderPass = new RenderPass(m_Device);
	const auto attachmentIdx = m_RenderPass->addColorAttachment(m_SwapChain->getFormat(), vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore,
																vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::SampleCountFlagBits::e1,
																vk::ImageLayout::eColorAttachmentOptimal, vk::ImageLayout::ePresentSrcKHR);

	Subpass subpass{};
	subpass.bindPoint = vk::PipelineBindPoint::eGraphics;
	subpass.attachments.emplace_back(attachmentIdx);
	m_RenderPass->addSubpass(subpass);

	m_RenderPass->finalize();
}

void VkContext::createFrameBuffers()
{
	const auto &buffers = m_SwapChain->getBuffers();

	m_Framebuffers = std::vector<vk::Framebuffer>(buffers.size());

	for (size_t i = 0; i < buffers.size(); i++)
	{
		const auto &buffer = buffers.at(i);

		vk::ImageView attachments[] = {buffer.view};
		vk::FramebufferCreateInfo framebufferCreateInfo{};
		framebufferCreateInfo.renderPass = *m_RenderPass;
		framebufferCreateInfo.attachmentCount = 1;
		framebufferCreateInfo.pAttachments = attachments;
		framebufferCreateInfo.width = m_SwapChain->getExtent().width;
		framebufferCreateInfo.height = m_SwapChain->getExtent().height;
		framebufferCreateInfo.layers = 1;

		m_Framebuffers.at(i) = m_Device->createFramebuffer(framebufferCreateInfo);
	}
}

void VkContext::createGraphicsPipeline()
{
	auto vertShader = ShaderModule(m_Device, "vkshaders/simple.vert.spv");
	auto fragShader = ShaderModule(m_Device, "vkshaders/simple.frag.spv");

	vk::PipelineVertexInputStateCreateInfo vertexInputInfo{};
	vertexInputInfo.setPVertexBindingDescriptions(nullptr);
	vertexInputInfo.setVertexAttributeDescriptionCount(0u);
	vertexInputInfo.setPVertexAttributeDescriptions(nullptr);

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
	inputAssembly.setTopology(vk::PrimitiveTopology::eTriangleList);
	inputAssembly.setPrimitiveRestartEnable(false);

	vk::Viewport viewport{};
	viewport.setX(0.0f);
	viewport.setY(0.0f);
	viewport.setWidth(static_cast<float>(m_SwapChain->getExtent().width));
	viewport.setHeight(static_cast<float>(m_SwapChain->getExtent().height));
	viewport.setMinDepth(0.0f);
	viewport.setMaxDepth(1.0f);

	vk::Rect2D scissor{};
	scissor.setOffset(vk::Offset2D(0, 0));
	scissor.setExtent(m_SwapChain->getExtent());

	vk::PipelineViewportStateCreateInfo viewportState{};
	viewportState.setViewportCount(1u);
	viewportState.setPViewports(&viewport);
	viewportState.setScissorCount(1u);
	viewportState.setPScissors(&scissor);

	vk::PipelineRasterizationStateCreateInfo rasterizer{};
	// If enabled then all geometry that's further away than the maximum depth
	//  is clamped to the maximum depth (useful for shadow maps)
	rasterizer.setDepthClampEnable(false);

	// If enabled then the geometry is never processed through this stage
	rasterizer.setRasterizerDiscardEnable(false);
	rasterizer.setPolygonMode(vk::PolygonMode::eFill);
	rasterizer.setLineWidth(1.0f);
	rasterizer.setCullMode(vk::CullModeFlagBits::eBack);
	rasterizer.setFrontFace(vk::FrontFace::eClockwise);
	rasterizer.setDepthBiasEnable(false);
	rasterizer.setDepthBiasConstantFactor(0.0f);
	rasterizer.setDepthBiasClamp(0.0f);
	rasterizer.setDepthBiasSlopeFactor(0.0f);

	vk::PipelineMultisampleStateCreateInfo multisampling{};
	multisampling.setSampleShadingEnable(false);
	multisampling.setRasterizationSamples(vk::SampleCountFlagBits::e1);
	multisampling.setMinSampleShading(1.0f);
	multisampling.setPSampleMask(nullptr);
	multisampling.setAlphaToCoverageEnable(false);
	multisampling.setAlphaToOneEnable(false);

	vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
	colorBlendAttachment.setColorWriteMask(vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
										   vk::ColorComponentFlagBits::eA);
	colorBlendAttachment.setBlendEnable(false);
	colorBlendAttachment.setSrcColorBlendFactor(vk::BlendFactor::eOne);
	colorBlendAttachment.setDstColorBlendFactor(vk::BlendFactor::eZero);
	colorBlendAttachment.setColorBlendOp(vk::BlendOp::eAdd);
	colorBlendAttachment.setSrcAlphaBlendFactor(vk::BlendFactor::eOne);
	colorBlendAttachment.setDstAlphaBlendFactor(vk::BlendFactor::eZero);
	colorBlendAttachment.setAlphaBlendOp(vk::BlendOp::eAdd);

	vk::PipelineColorBlendStateCreateInfo colorBlending{};
	colorBlending.setLogicOpEnable(false);
	colorBlending.setLogicOp(vk::LogicOp::eCopy);
	colorBlending.setAttachmentCount(1u);
	colorBlending.setPAttachments(&colorBlendAttachment);
	colorBlending.setBlendConstants({0.0f, 0.0f, 0.0f, 0.0f});

	std::array<vk::DynamicState, 2> dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eLineWidth};
	vk::PipelineDynamicStateCreateInfo dynamicState{};
	dynamicState.setDynamicStateCount(static_cast<uint32_t>(dynamicStates.size()));
	dynamicState.setPDynamicStates(dynamicStates.data());

	vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
	pipelineLayoutInfo.setSetLayoutCount(0u);
	pipelineLayoutInfo.setPSetLayouts(nullptr);
	pipelineLayoutInfo.setPushConstantRangeCount(0u);
	pipelineLayoutInfo.setPPushConstantRanges(nullptr);

	m_PipelineLayout = m_Device->createPipelineLayout(pipelineLayoutInfo);
	assert(m_PipelineLayout);

	std::array<vk::PipelineShaderStageCreateInfo, 2> stages = {vertShader.getShaderStage(vk::ShaderStageFlagBits::eVertex),
															   fragShader.getShaderStage(vk::ShaderStageFlagBits::eFragment)};

	vk::GraphicsPipelineCreateInfo pipelineInfo{};
	pipelineInfo.setStageCount(static_cast<uint32_t>(stages.size()));
	pipelineInfo.setPStages(stages.data());
	pipelineInfo.setPVertexInputState(&vertexInputInfo);
	pipelineInfo.setPInputAssemblyState(&inputAssembly);
	pipelineInfo.setPViewportState(&viewportState);
	pipelineInfo.setPRasterizationState(&rasterizer);
	pipelineInfo.setPMultisampleState(&multisampling);
	pipelineInfo.setPDepthStencilState(nullptr);
	pipelineInfo.setPColorBlendState(&colorBlending);
	pipelineInfo.setPDynamicState(nullptr);
	pipelineInfo.setLayout(m_PipelineLayout);
	pipelineInfo.setRenderPass(*m_RenderPass);
	pipelineInfo.setSubpass(0u);
	pipelineInfo.setBasePipelineHandle(nullptr);
	pipelineInfo.setBasePipelineIndex(-1);

	m_GraphicsPipeline = m_Device->createGraphicsPipeline(nullptr, pipelineInfo, nullptr);
	assert(m_GraphicsPipeline);
}

void VkContext::printAvailableExtensions() const
{
	// Get supported extensions
	const auto availableExtensions = vk::enumerateInstanceExtensionProperties();

	if (!availableExtensions.empty())
	{
		utils::logger::log("VkContext", "Available extensions:");
		for (const auto &extension : availableExtensions)
		{
			std::cout << "\t" << extension.extensionName << std::endl;
		}
	}
}

void VkContext::createCommandBuffers()
{
	const auto buffers = m_SwapChain->getBuffers();
	m_CommandBuffers = m_Device.createCommandBuffers(buffers.size(), vk::CommandBufferLevel::ePrimary, VulkanDevice::GRAPHICS);

	vk::CommandBufferBeginInfo beginInfo = {};
	vk::ClearValue clearColor{};
	clearColor.color.float32[0] = 0.0f;
	clearColor.color.float32[1] = 0.0f;
	clearColor.color.float32[2] = 0.0f;
	clearColor.color.float32[3] = 1.0f;

	for (size_t i = 0; i < buffers.size(); i++)
	{
		const auto &buffer = buffers.at(i);
		const auto &cmdBuffer = m_CommandBuffers.at(i);

		cmdBuffer.begin(beginInfo);

		vk::ImageMemoryBarrier imageBarrier =
			vk::ImageMemoryBarrier({}, {}, vk::ImageLayout::eGeneral, vk::ImageLayout::eColorAttachmentOptimal, m_Device.getQueueIndices().graphicsIdx.value(),
								   m_Device.getQueueIndices().graphicsIdx.value(), buffer.image);
		cmdBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics, {}, {}, {}, {imageBarrier});

		vk::RenderPassBeginInfo renderPassInfo =
			vk::RenderPassBeginInfo(*m_RenderPass, m_Framebuffers.at(i), {{0, 0}, m_SwapChain->getExtent()}, 1, &clearColor);

		cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline);
		cmdBuffer.draw(3, 1, 0, 0);
		cmdBuffer.endRenderPass();
		cmdBuffer.end();
	}
}

void VkContext::createSemaphores()
{
	const auto count = uint32_t(m_SwapChain->getBuffers().size());

	vk::SemaphoreCreateInfo createInfo{};
	m_ImageAvailableSemaphores.resize(count);
	m_RenderFinishedSemaphores.resize(count);

	for (uint32_t i = 0; i < count; i++)
	{
		m_ImageAvailableSemaphores.at(i) = m_Device->createSemaphore(createInfo);
		m_RenderFinishedSemaphores.at(i) = m_Device->createSemaphore(createInfo);
	}
}

void vkc::VkContext::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void vkc::VkContext::setTextures(const std::vector<rfw::TextureData> &materials) {}

void vkc::VkContext::setMesh(size_t index, const Mesh &mesh)
{
	return;
	if (m_Meshes.size() >= index)
	{
		while (m_Meshes.size() >= index)
			m_Meshes.push_back(new vkc::VkMesh(m_Device));
	}

	m_Meshes.at(index)->setGeometry(mesh);
}

void vkc::VkContext::setInstance(size_t instanceIdx, size_t mesh, const mat4 &transform, const mat3 &inverse_transform) {}

void vkc::VkContext::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

void vkc::VkContext::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
							   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights)
{
}

void VkContext::getProbeResults(unsigned *instanceIndex, unsigned *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings VkContext::getAvailableSettings() const { return rfw::AvailableRenderSettings(); }

void VkContext::setSetting(const rfw::RenderSetting &setting) {}

void vkc::VkContext::update()
{
	return;
	m_VertexBuffers.resize(m_Meshes.size());
	for (size_t i = 0; i < m_Meshes.size(); i++)
		m_VertexBuffers.at(i) = *m_Meshes.at(i)->vertices;

	vk::ClearValue clearColor{};
	clearColor.color.float32[0] = 0.0f;
	clearColor.color.float32[1] = 0.0f;
	clearColor.color.float32[2] = 0.0f;
	clearColor.color.float32[3] = 1.0f;

	for (size_t i = 0; i < m_CommandBuffers.size(); i++)
	{
		auto cmdBuffer = m_CommandBuffers.at(i);
		cmdBuffer.begin(vk::CommandBufferBeginInfo());

		vk::RenderPassBeginInfo renderPassInfo =
			vk::RenderPassBeginInfo(*m_RenderPass, m_Framebuffers.at(i), {{0, 0}, m_SwapChain->getExtent()}, 1, &clearColor);

		cmdBuffer.beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
		cmdBuffer.bindVertexBuffers(0, m_VertexBuffers.size(), m_VertexBuffers.data(), nullptr);
		cmdBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, m_GraphicsPipeline);
		cmdBuffer.draw(3, 1, 0, 0);
		cmdBuffer.endRenderPass();
		cmdBuffer.end();
	}
}

void vkc::VkContext::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats vkc::VkContext::getStats() const { return rfw::RenderStats(); }

rfw::RenderContext *createRenderContext() { return new vkc::VkContext(); }
void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }
