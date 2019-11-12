#include "Device.h"

#include <set>
#include <string>
#include <utils/Logger.h>

using namespace vkc;

QueueFamilyIndices::QueueFamilyIndices(vk::PhysicalDevice device, vk::SurfaceKHR surface)
{
	const auto queueFamilies = device.getQueueFamilyProperties();
	uint32_t i = 0u;
	for (const auto &qf : queueFamilies)
	{
		if (qf.queueCount > 0)
		{
			if (!this->graphics.has_value() && qf.queueFlags & vk::QueueFlagBits::eGraphics)
				this->graphics = i;
			if (!this->compute.has_value() && qf.queueFlags & vk::QueueFlagBits::eCompute)
				this->compute = i;
			if (!this->transfer.has_value() && qf.queueFlags & vk::QueueFlagBits::eTransfer)
				this->transfer = i;
			if (!this->present.has_value() && device.getSurfaceSupportKHR(i, surface))
				this->present = i;
		}
		i++;
	}
}

Device::Device(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
{
	m_Members = std::make_shared<DeviceMembers>(DeviceMembers(physicalDevice, surface));
	const float priority = 1.0f;

	vk::PhysicalDeviceFeatures deviceFeatures{};

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
	std::set<uint32_t> uniqueQueueFamilies = {m_Members->m_Indices.graphics.value(),
											  m_Members->m_Indices.compute.value(),
											  m_Members->m_Indices.present.value()};

	for (auto queueFamily : uniqueQueueFamilies)
	{
		vk::DeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.setQueueFamilyIndex(queueFamily);
		queueCreateInfo.setQueueCount(1u);
		queueCreateInfo.setPQueuePriorities(&priority);
		queueCreateInfos.push_back(queueCreateInfo);
	}

	const auto properties = physicalDevice.getProperties();

	const auto deviceExtensions = Device::getDeviceExtensions();

	vk::DeviceCreateInfo createInfo =
		vk::DeviceCreateInfo({}, uint32_t(queueCreateInfos.size()), queueCreateInfos.data(), 0, nullptr,
							 uint32_t(deviceExtensions.size()), deviceExtensions.data(), &deviceFeatures);

	// Since Vulkan >=1.1 validation layers are automatically enabled when they are enabled during context creation
	createInfo.setEnabledLayerCount(0u);
	createInfo.setPpEnabledLayerNames(nullptr);

	m_Members->m_Device = m_Members->m_PhysicalDevice.createDevice(createInfo);
	assert(m_Members->m_Device);
	if (!m_Members->m_Device)
		throw std::runtime_error("Failed to create logical device!");

	m_Members->m_GraphicsQueue = m_Members->m_Device.getQueue(m_Members->m_Indices.graphics.value(), 0);
	m_Members->m_ComputeQueue = m_Members->m_Device.getQueue(m_Members->m_Indices.compute.value(), 0);
	m_Members->m_PresentQueue = m_Members->m_Device.getQueue(m_Members->m_Indices.present.value(), 0);
	m_Members->m_TransferQueue = m_Members->m_Device.getQueue(m_Members->m_Indices.transfer.value(), 0);

	m_Members->m_MemProps = m_Members->m_PhysicalDevice.getMemoryProperties();

	vk::CommandPoolCreateInfo commandPoolCreateInfo{};
	commandPoolCreateInfo.setPNext(nullptr);
	commandPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
	commandPoolCreateInfo.setQueueFamilyIndex(m_Members->m_Indices.graphics.value());
	m_Members->m_CommandPool = m_Members->m_Device.createCommandPool(commandPoolCreateInfo);

	char buffer[1024];
	sprintf(buffer, "Render device \"%s\" successfully initialized.", properties.deviceName);
	DEBUG(buffer);
}

vk::Queue Device::getGraphicsQueue() const { return m_Members->m_GraphicsQueue; }

vk::Queue Device::getComputeQueue() const { return m_Members->m_ComputeQueue; }

vk::Queue Device::getTransferQueue() const { return m_Members->m_TransferQueue; }

vk::Queue Device::getPresentQueue() const { return m_Members->m_PresentQueue; }

vk::Device Device::getVkDevice() const { return m_Members->m_Device; }

vk::PhysicalDevice Device::getVkPhysicalDevice() const { return m_Members->m_PhysicalDevice; }

const QueueFamilyIndices &Device::getQueueFamilyIndices() const { return m_Members->m_Indices; }

vk::CommandPool Device::getCommandPool() const { return m_Members->m_CommandPool; }

bool Device::checkExtensionSupport(vk::PhysicalDevice physicalDevice)
{
	const auto deviceExtensions = Device::getDeviceExtensions();
	const auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
	std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());

	// Check if all required extensions are available
	for (const auto &extension : availableExtensions)
		requiredExtensions.erase(extension.extensionName);

	return requiredExtensions.empty();
}

unsigned Device::rateDevice(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface)
{
	unsigned int score = 0;

	const QueueFamilyIndices indices(physicalDevice, surface);
	if (!indices.isComplete() || !checkExtensionSupport(physicalDevice))
		return score;

	const auto deviceProperties = physicalDevice.getProperties();
	const auto deviceFeatures = physicalDevice.getFeatures();

	if (deviceProperties.deviceType == vk::PhysicalDeviceType::eCpu)
		score += 10;
	else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
		score += 100;
	else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		score += 1000;

	score += deviceProperties.limits.maxImageDimension2D;

	return score;
}

std::vector<const char *> Device::getDeviceExtensions() { return {VK_KHR_SWAPCHAIN_EXTENSION_NAME}; }

uint32_t Device::getMemoryType(const vk::MemoryRequirements &memReqs, const vk::MemoryPropertyFlags &memProps) const
{
	for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < VK_MAX_MEMORY_TYPES; ++memoryTypeIndex)
	{
		if (memReqs.memoryTypeBits & (1u << memoryTypeIndex))
			if (m_Members->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags & memProps)
				return memoryTypeIndex;
	}

	return 0;
}

void Device::cleanup()
{
	if (m_Members)
		m_Members.reset();
}

Device::Device(const Device &rhs) { m_Members = rhs.m_Members; }

Device::DeviceMembers::~DeviceMembers()
{
	if (m_Device)
	{
		m_Device.waitIdle();
		if (m_CommandPool)
		{
			m_Device.destroyCommandPool(m_CommandPool);
			m_CommandPool = nullptr;
		}
		m_Device.destroy();
		m_Device = nullptr;
	}
}
