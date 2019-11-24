#pragma once

#include <vulkan/vulkan.hpp>
#include <optional>
#include <unordered_set>

#include <vk_mem_alloc.h>

namespace vkc
{
template<typename T> class VmaBuffer;
struct QueueFamilyIndices
{
	QueueFamilyIndices() = default;
	QueueFamilyIndices(vk::PhysicalDevice device, std::optional<vk::SurfaceKHR> surface = std::nullopt);
	bool IsComplete() const;
	std::unordered_set<uint32_t> getUniqueQueueIds() const;

	bool hasSurface = false;
	std::optional<uint32_t> graphicsIdx;
	std::optional<uint32_t> computeIdx;
	std::optional<uint32_t> transferIdx;
	std::optional<uint32_t> presentIdx;
};

class OneTimeCommandBuffer;

/*
 * This class is a wrapper around vk::Device with some helper functions.
 * The object is reference counted to make using a Vulkan device easier.
 * It also allows objects to keep a relatively cheap reference as this object's
 *  members are stored in a reference counted pointer.
 */
class VulkanDevice
{
  public:
	enum QueueType
	{
		COMPUTE = 0,
		GRAPHICS = 1,
		TRANSFER = 2,
		PRESENT = 3,
		QUEUE_TYPE_COUNT = 4
	};

  private:
	struct DeviceMembers
	{
		DeviceMembers(vk::PhysicalDevice &pDevice, std::optional<vk::SurfaceKHR> &surface)
			: m_Indices(pDevice, surface), m_PhysicalDevice(pDevice)
		{
		}

		~DeviceMembers() { cleanup(); }

		void cleanup()
		{
			if (m_VkDevice)
			{
				m_VkDevice.waitIdle();
				for (int i = 0; i < m_CommandPools.size(); i++)
					if (m_CommandPools.at(i))
						m_VkDevice.destroyCommandPool(m_CommandPools.at(i));

				m_VkDevice.destroy();
				m_VkDevice = nullptr;
			}
			if (allocator)
			{
				vmaDestroyAllocator(allocator);
				allocator = nullptr;
			}
		}

		QueueFamilyIndices m_Indices;
		std::vector<vk::CommandPool> m_CommandPools;
		vk::PhysicalDevice m_PhysicalDevice;
		vk::PhysicalDeviceMemoryProperties m_MemProps;
		vk::Device m_VkDevice;
		vk::Queue m_GraphicsQueue;
		vk::Queue m_ComputeQueue;
		vk::Queue m_TransferQueue;
		vk::Queue m_PresentQueue;
		vk::DispatchLoaderDynamic m_DynamicDispatcher;
		VmaAllocator allocator;
		uint32_t queueCmdPoolIndices[QUEUE_TYPE_COUNT];
	};

  public:
	VulkanDevice() = default;
	VulkanDevice(const VulkanDevice &rhs);
	VulkanDevice(vk::Instance instance, vk::PhysicalDevice physicalDevice, const std::vector<const char *> &extensions,
				 std::optional<vk::SurfaceKHR> surface = std::nullopt);
	~VulkanDevice() = default;

	[[nodiscard]] const QueueFamilyIndices &getQueueIndices() const { return m_Members->m_Indices; }
	vk::Queue getQueue(QueueType type) const
	{
		switch (type)
		{
		case (COMPUTE):
			return getComputeQueue();
		case (GRAPHICS):
			return getGraphicsQueue();
		case (TRANSFER):
			return getTransferQueue();
		case (PRESENT):
			return getPresentQueue();
		default:
			break;
		}

		return nullptr;
	}

	vk::Queue getGraphicsQueue() const { return m_Members->m_GraphicsQueue; }
	vk::Queue getComputeQueue() const { return m_Members->m_ComputeQueue; }
	vk::Queue getTransferQueue() const { return m_Members->m_TransferQueue; }
	vk::Queue getPresentQueue() const { return m_Members->m_PresentQueue; }

	[[nodiscard]] vk::Device getVkDevice() const { return m_Members->m_VkDevice; }
	[[nodiscard]] vk::PhysicalDevice getPhysicalDevice() const { return m_Members->m_PhysicalDevice; }
	[[nodiscard]] vk::PhysicalDeviceMemoryProperties getMemoryProperties() const { return m_Members->m_MemProps; }
	[[nodiscard]] vk::CommandPool getCommandPool(QueueType type) const
	{
		return m_Members->m_CommandPools.at(m_Members->queueCmdPoolIndices[type]);
	}
	[[nodiscard]] uint32_t getMemoryType(const vk::MemoryRequirements &memReqs, vk::MemoryPropertyFlags memProps) const;
	void cleanup();

	vk::CommandBuffer createCommandBuffer(vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
										  QueueType type = GRAPHICS);

	std::vector<vk::CommandBuffer> createCommandBuffers(uint32_t count,
														vk::CommandBufferLevel level = vk::CommandBufferLevel::ePrimary,
														QueueType type = GRAPHICS);
	OneTimeCommandBuffer createOneTimeCmdBuffer(vk::CommandBufferLevel = vk::CommandBufferLevel::ePrimary,
												QueueType type = GRAPHICS);
	void submitCommandBuffer(vk::CommandBuffer cmdBuffer, vk::Queue queue, vk::Fence fence = nullptr,
							 vk::PipelineStageFlags waitStageMask = {}, uint32_t waitSemaphoreCount = 0,
							 vk::Semaphore *waitSemaphores = nullptr, uint32_t signalSemaphoreCount = 0,
							 vk ::Semaphore *signalSemaphores = nullptr);
	void submitCommandBuffers(uint32_t cmdBufferCount, vk::CommandBuffer *cmdBuffer, vk::Queue queue,
							  vk::Fence fence = nullptr, vk::PipelineStageFlags waitStageMask = {},
							  uint32_t waitSemaphoreCount = 0, vk::Semaphore *waitSemaphores = nullptr,
							  uint32_t signalSemaphoreCount = 0, vk::Semaphore *signalSemaphores = nullptr);
	void freeCommandBuffer(vk::CommandBuffer cmdBuffer, QueueType type = GRAPHICS);
	void freeCommandBuffers(const std::vector<vk::CommandBuffer> &cmdBuffers, QueueType type = GRAPHICS);
	void waitIdle() const;
	vk::DispatchLoaderDynamic &getLoader() { return m_Members->m_DynamicDispatcher; }

	vk::Device *operator->() { return &m_Members->m_VkDevice; }

	static std::optional<vk::PhysicalDevice>
	pickDeviceWithExtensions(vk::Instance &instance, const std::vector<const char *> &extensions,
							 std::optional<vk::SurfaceKHR> surface = std::nullopt);

	static unsigned int rateDevice(const vk::PhysicalDevice &pDevice,
								   std::optional<vk::SurfaceKHR> surface = std::nullopt);

	template <typename T> VmaBuffer<T> createBuffer() { return VmaBuffer<T>(*this); }

	VmaAllocator getAllocator() const { return m_Members->allocator; }

	operator VkDevice() { return m_Members->m_VkDevice; }
	operator VkPhysicalDevice() { return m_Members->m_PhysicalDevice; }
	operator vk::Device() { return m_Members->m_VkDevice; }
	operator vk::PhysicalDevice() { return m_Members->m_PhysicalDevice; }
	operator bool() { return m_Members != nullptr && m_Members->m_VkDevice; }

  private:
	std::shared_ptr<DeviceMembers> m_Members;
};

class OneTimeCommandBuffer
{
  public:
	OneTimeCommandBuffer(const VulkanDevice &device, vk::CommandBuffer cmdBuffer)
	{
		m_Recording = false;
		m_Device = device;
		m_CmdBuffer = cmdBuffer;
		begin();
	}

	~OneTimeCommandBuffer() { cleanup(); }

	void cleanup()
	{
		if (m_CmdBuffer)
			m_Device.freeCommandBuffer(m_CmdBuffer);
	}

	void begin()
	{
		assert(!m_Recording);
		m_Recording = true;
		m_CmdBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
	}

	void end()
	{
		assert(m_Recording);
		m_CmdBuffer.end();
		m_Recording = false;
	}

	void submit(vk::Queue queue, bool wait = true)
	{
		if (m_Recording)
			end(), m_Recording = false;
		m_Device.submitCommandBuffer(m_CmdBuffer, queue);
		if (wait)
			queue.waitIdle();
	}

	vk::CommandBuffer getVkCommandBuffer() const { return m_CmdBuffer; }

	vk::CommandBuffer *operator->() { return &m_CmdBuffer; }
	operator vk::CommandBuffer() { return m_CmdBuffer; }

  private:
	bool m_Recording;
	VulkanDevice m_Device;
	vk::CommandBuffer m_CmdBuffer;
};
} // namespace vkrtx
