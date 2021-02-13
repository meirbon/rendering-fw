#include "Context.h"

using namespace vkrtx;

QueueFamilyIndices::QueueFamilyIndices(vk::PhysicalDevice device, std::optional<vk::SurfaceKHR> surface) : hasSurface(surface.has_value())
{
	const auto queueFamilies = device.getQueueFamilyProperties();

	auto i = 0u;
	for (const auto &qf : queueFamilies)
	{
		if (qf.queueCount > 0)
		{
			if (!this->graphicsIdx.has_value() && qf.queueFlags & vk::QueueFlagBits::eGraphics)
			{
				if (surface.has_value())
				{
					if (device.getSurfaceSupportKHR(i, surface.value()))
						this->graphicsIdx = i, this->presentIdx = i;
				}
				else
					this->graphicsIdx = i;
			}

			if (!this->computeIdx.has_value() && qf.queueFlags & vk::QueueFlagBits::eCompute)
				this->computeIdx = i;
			if (!this->transferIdx.has_value() && qf.queueFlags & vk::QueueFlagBits::eTransfer)
				this->transferIdx = i;
		}
		++i;
	}
}

bool QueueFamilyIndices::IsComplete() const
{
	return graphicsIdx.has_value() && computeIdx.has_value() && transferIdx.has_value() && hasSurface ? presentIdx.has_value() : true;
}

std::unordered_set<uint32_t> vkrtx::QueueFamilyIndices::getUniqueQueueIds() const
{
	std::vector<uint32_t> ids;
	ids.reserve(4);

	if (this->computeIdx.has_value())
		ids.push_back(this->computeIdx.value());

	if (this->graphicsIdx.has_value())
		ids.push_back(this->graphicsIdx.value());

	if (this->transferIdx.has_value())
		ids.push_back(this->transferIdx.value());

	if (this->presentIdx.has_value())
		ids.push_back(this->presentIdx.value());

	return std::unordered_set<uint32_t>(ids.data(), ids.data() + ids.size());
}

VulkanDevice::VulkanDevice(const VulkanDevice &rhs) : m_Members(rhs.m_Members) {}

VulkanDevice::VulkanDevice(vk::Instance instance, vk::PhysicalDevice physicalDevice, const std::vector<const char *> &extensions,
						   std::optional<vk::SurfaceKHR> surface)
{
	m_Members = std::make_shared<DeviceMembers>(physicalDevice, surface);

	const float priority = 1.0f;

	const auto deviceFeatures = physicalDevice.getFeatures();

	vk::PhysicalDeviceDescriptorIndexingFeaturesEXT indexingFeatures{};
	indexingFeatures.pNext = nullptr;
	vk::PhysicalDeviceFeatures2 deviceFeatures2{};
	deviceFeatures2.pNext = &indexingFeatures;
	physicalDevice.getFeatures2(&deviceFeatures2);
	if (!indexingFeatures.runtimeDescriptorArray || !indexingFeatures.descriptorBindingPartiallyBound ||
		!indexingFeatures.descriptorBindingVariableDescriptorCount)
		FAILURE("Device does not support runtimeDescriptorArray, descriptorBindingPartiallyBound or "
				"descriptorBindingVariableDescriptorCount  which are needed for the Vulkan render core.");
	indexingFeatures.runtimeDescriptorArray = true; // Enable feature
	indexingFeatures.descriptorBindingPartiallyBound = true;
	indexingFeatures.descriptorBindingVariableDescriptorCount = true;

	std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos{};
	std::set<uint32_t> uniqueQueueFamilies = {m_Members->m_Indices.graphicsIdx.value(), m_Members->m_Indices.computeIdx.value(),
											  m_Members->m_Indices.transferIdx.value(),
											  surface.has_value() ? m_Members->m_Indices.presentIdx.value() : m_Members->m_Indices.graphicsIdx.value()};

	uint i = 0;
	for (const auto qfIdx : uniqueQueueFamilies)
	{
		vk::DeviceQueueCreateInfo queueCreateInfo{};
		queueCreateInfo.setQueueFamilyIndex(qfIdx);
		queueCreateInfo.setQueueCount(1u);
		queueCreateInfo.setPQueuePriorities(&priority);
		queueCreateInfos.push_back(queueCreateInfo);
	}

	vk::DeviceCreateInfo createInfo{};
	createInfo.setPNext(&indexingFeatures);
	createInfo.setPQueueCreateInfos(queueCreateInfos.data());
	createInfo.setQueueCreateInfoCount(static_cast<uint32_t>(queueCreateInfos.size()));
	createInfo.setPEnabledFeatures(&deviceFeatures);
	createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()));
	createInfo.setPpEnabledExtensionNames(extensions.data());
	createInfo.setEnabledLayerCount(0u);
	createInfo.setPpEnabledLayerNames(nullptr);

	m_Members->m_VkDevice = m_Members->m_PhysicalDevice.createDevice(createInfo);
	if (!m_Members->m_VkDevice)
		FAILURE("Could not initialize Vulkan device.");

	m_Members->m_GraphicsQueue = m_Members->m_VkDevice.getQueue(m_Members->m_Indices.graphicsIdx.value(), 0);
	m_Members->m_ComputeQueue = m_Members->m_VkDevice.getQueue(m_Members->m_Indices.computeIdx.value(), 0);
	m_Members->m_TransferQueue = m_Members->m_VkDevice.getQueue(m_Members->m_Indices.transferIdx.value(), 0);

	if (surface.has_value())
		m_Members->m_PresentQueue = m_Members->m_VkDevice.getQueue(m_Members->m_Indices.presentIdx.value(), 0);

	m_Members->m_MemProps = m_Members->m_PhysicalDevice.getMemoryProperties();

	vk::CommandPoolCreateInfo cmdPoolCreateInfo{};
	cmdPoolCreateInfo.setPNext(nullptr);
	cmdPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);

	const auto ids = m_Members->m_Indices.getUniqueQueueIds();
	m_Members->m_CommandPools.resize(QUEUE_TYPE_COUNT);

	m_Members->queueCmdPoolIndices[COMPUTE] = 0;
	m_Members->queueCmdPoolIndices[GRAPHICS] = 0;
	m_Members->queueCmdPoolIndices[TRANSFER] = 0;
	m_Members->queueCmdPoolIndices[PRESENT] = 0;

	if (m_Members->m_Indices.computeIdx.has_value())
		m_Members->queueCmdPoolIndices[COMPUTE] = m_Members->m_Indices.computeIdx.value();
	if (m_Members->m_Indices.graphicsIdx.has_value())
		m_Members->queueCmdPoolIndices[GRAPHICS] = m_Members->m_Indices.graphicsIdx.value();
	if (m_Members->m_Indices.transferIdx.has_value())
		m_Members->queueCmdPoolIndices[TRANSFER] = m_Members->m_Indices.transferIdx.value();
	if (m_Members->m_Indices.presentIdx.has_value())
		m_Members->queueCmdPoolIndices[PRESENT] = m_Members->m_Indices.presentIdx.value();

	i = 0;
	for (const auto id : ids)
	{
		vk::CommandPoolCreateInfo cmdPoolCreateInfo{};
		cmdPoolCreateInfo.setPNext(nullptr);
		cmdPoolCreateInfo.setFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer);
		cmdPoolCreateInfo.setQueueFamilyIndex(id);
		m_Members->m_CommandPools.at(i) = m_Members->m_VkDevice.createCommandPool(cmdPoolCreateInfo);
		i++;
	}

	printf("Vulkan device %s initialized.\n", physicalDevice.getProperties().deviceName.data());

	m_Members->m_DynamicDispatcher.init(instance, m_Members->m_VkDevice);

	VmaAllocatorCreateInfo allocCreateInfo{};
	allocCreateInfo.physicalDevice = (VkPhysicalDevice)physicalDevice;
	allocCreateInfo.device = (VkDevice)m_Members->m_VkDevice;
	allocCreateInfo.flags |= VMA_ALLOCATOR_CREATE_KHR_DEDICATED_ALLOCATION_BIT;
	// More memory usage, but provides better performance
	// allocCreateInfo.flags = VmaPoolCreateFlagBits::VMA_POOL_CREATE_LINEAR_ALGORITHM_BIT;
	vmaCreateAllocator(&allocCreateInfo, &m_Members->allocator);
}

uint32_t VulkanDevice::getMemoryType(const vk::MemoryRequirements &memReqs, vk::MemoryPropertyFlags memProps) const
{
	for (uint32_t memoryTypeIndex = 0; memoryTypeIndex < VK_MAX_MEMORY_TYPES; ++memoryTypeIndex)
	{
		if (memReqs.memoryTypeBits & (1u << memoryTypeIndex))
			if (m_Members->m_MemProps.memoryTypes[memoryTypeIndex].propertyFlags & memProps)
				return memoryTypeIndex;
	}

	return 0;
}

void VulkanDevice::cleanup()
{
	if (m_Members)
		m_Members->cleanup();
}

vk::CommandBuffer VulkanDevice::createCommandBuffer(vk::CommandBufferLevel level, QueueType type)
{
	vk::CommandBufferAllocateInfo commandBufferAllocInfo{};
	commandBufferAllocInfo.setPNext(nullptr);
	commandBufferAllocInfo.setCommandPool(m_Members->m_CommandPools.at(m_Members->queueCmdPoolIndices[type]));
	commandBufferAllocInfo.setLevel(level);
	commandBufferAllocInfo.setCommandBufferCount(1);
	
	vk::CommandBuffer cmdBuffer;
	CheckVK(m_Members->m_VkDevice.allocateCommandBuffers(&commandBufferAllocInfo, &cmdBuffer));
	return cmdBuffer;
}

std::vector<vk::CommandBuffer> VulkanDevice::createCommandBuffers(uint32_t count, vk::CommandBufferLevel level, QueueType type)
{
	vk::CommandBufferAllocateInfo commandBufferAllocInfo{};
	commandBufferAllocInfo.setPNext(nullptr);
	commandBufferAllocInfo.setCommandPool(m_Members->m_CommandPools.at(m_Members->queueCmdPoolIndices[type]));
	commandBufferAllocInfo.setLevel(level);
	commandBufferAllocInfo.setCommandBufferCount(count);

	return m_Members->m_VkDevice.allocateCommandBuffers(commandBufferAllocInfo);
}

OneTimeCommandBuffer VulkanDevice::createOneTimeCmdBuffer(vk::CommandBufferLevel level, QueueType type)
{
	auto commandBuffer = createCommandBuffer(level, type);

	return OneTimeCommandBuffer(*this, commandBuffer);
}

void VulkanDevice::submitCommandBuffer(vk::CommandBuffer cmdBuffer, vk::Queue queue, vk::Fence fence, vk::PipelineStageFlags waitStageMask,
									   uint32_t waitSemaphoreCount, vk::Semaphore *waitSemaphores, uint32_t signalSemaphoreCount,
									   vk ::Semaphore *signalSemaphores)
{
	// submit build command to queue
	vk::SubmitInfo submitInfo = vk::SubmitInfo(waitSemaphoreCount, waitSemaphores, &waitStageMask, 1, &cmdBuffer, signalSemaphoreCount, signalSemaphores);
	queue.submit({submitInfo}, fence);
}

void VulkanDevice::submitCommandBuffers(uint32_t cmdBufferCount, vk::CommandBuffer *cmdBuffers, vk::Queue queue, vk::Fence fence,
										vk::PipelineStageFlags waitStageMask, uint32_t waitSemaphoreCount, vk::Semaphore *waitSemaphores,
										uint32_t signalSemaphoreCount, vk::Semaphore *signalSemaphores)
{
	vk::SubmitInfo submitInfo =
		vk::SubmitInfo(waitSemaphoreCount, waitSemaphores, &waitStageMask, cmdBufferCount, cmdBuffers, signalSemaphoreCount, signalSemaphores);
	queue.submit({submitInfo}, fence);
	queue.waitIdle();
}

void VulkanDevice::freeCommandBuffer(vk::CommandBuffer cmdBuffer, QueueType type)
{
	m_Members->m_VkDevice.freeCommandBuffers(m_Members->m_CommandPools.at(m_Members->queueCmdPoolIndices[type]), {cmdBuffer});
}

void VulkanDevice::freeCommandBuffers(const std::vector<vk::CommandBuffer> &cmdBuffers, QueueType type)
{
	m_Members->m_VkDevice.freeCommandBuffers(m_Members->m_CommandPools.at(m_Members->queueCmdPoolIndices[type]), cmdBuffers);
}

void VulkanDevice::waitIdle() const { m_Members->m_VkDevice.waitIdle(); }

std::optional<vk::PhysicalDevice> VulkanDevice::pickDeviceWithExtensions(vk::Instance &instance, const std::vector<const char *> &extensions,
																		 std::optional<vk::SurfaceKHR> surface)
{
	const auto physicalDevices = instance.enumeratePhysicalDevices();

	// Use multimap to sort devices based on score
	std::multimap<uint, vk::PhysicalDevice> candidates;

	// Retrieve score for every device
	for (const auto &pDevice : physicalDevices)
		candidates.insert(std::make_pair(VulkanDevice::rateDevice(pDevice, surface), pDevice));

	// Early out if we have no valid candidates
	if (candidates.empty())
		return std::nullopt;

	// Iterate over candidates till we find one that supports ray tracing
	for (const auto &candidate : candidates)
	{
		std::set<std::string> requiredExtensions(extensions.begin(), extensions.end());

		const auto dExtensions = candidate.second.enumerateDeviceExtensionProperties();
		for (const auto &ext : dExtensions)
			requiredExtensions.erase(ext.extensionName);

		// Device supports every requested extension
		if (requiredExtensions.empty())
			return std::make_optional(candidate.second);
	}

	// No supported device found
	return std::nullopt;
}

unsigned VulkanDevice::rateDevice(const vk::PhysicalDevice &pDevice, std::optional<vk::SurfaceKHR> surface)
{
	uint score = 0;

	// Check whether device supports our required queues
	const QueueFamilyIndices indices(pDevice, surface);
	if (!indices.IsComplete())
		return score;

	const auto deviceProperties = pDevice.getProperties();

	// Give fastest device highest score
	if (deviceProperties.deviceType == vk::PhysicalDeviceType::eCpu)
		score += 10;
	else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
		score += 100;
	else if (deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu)
		score += 1000;

	score += deviceProperties.limits.maxImageDimension2D;

	return score;
}
