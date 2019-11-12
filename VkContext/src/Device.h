#pragma once

#include <vulkan/vulkan.hpp>

#include <optional>
#include <memory>

namespace vkc
{
struct QueueFamilyIndices
{
	explicit QueueFamilyIndices(vk::PhysicalDevice device, vk::SurfaceKHR surface);
	[[nodiscard]] bool isComplete() const
	{
		return graphics.has_value() && compute.has_value() && present.has_value() && transfer.has_value();
	}

	std::optional<uint32_t> graphics;
	std::optional<uint32_t> compute;
	std::optional<uint32_t> present;
	std::optional<uint32_t> transfer;
};

/*
 * Reference counted instance of a Vulkan device using a shared_ptr as its members.
 * The shared_ptr makes sure a reference to our device does not make other objects
 * too much larger while maintaining flexibility in terms of functionality
 */
class Device
{
  public:
	Device() = default;
	Device(const Device &rhs);
	Device(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);
	~Device() = default;

	[[nodiscard]] uint32_t getMemoryType(const vk::MemoryRequirements &memReqs,
										 const vk::MemoryPropertyFlags &memProps) const;
	[[nodiscard]] vk::Queue getGraphicsQueue() const;
	[[nodiscard]] vk::Queue getComputeQueue() const;
	[[nodiscard]] vk::Queue getTransferQueue() const;
	[[nodiscard]] vk::Queue getPresentQueue() const;
	[[nodiscard]] vk::Device getVkDevice() const;
	[[nodiscard]] vk::PhysicalDevice getVkPhysicalDevice() const;
	[[nodiscard]] const QueueFamilyIndices &getQueueFamilyIndices() const;
	[[nodiscard]] vk::CommandPool getCommandPool() const;

	static bool checkExtensionSupport(vk::PhysicalDevice physicalDevice);
	static unsigned int rateDevice(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface);

	static std::vector<const char *> getDeviceExtensions();

	operator vk::PhysicalDevice() const { return m_Members->m_PhysicalDevice; }
	operator VkPhysicalDevice() const { return m_Members->m_PhysicalDevice; }

	operator vk::Device() const { return m_Members->m_Device; }
	operator VkDevice() const { return m_Members->m_Device; }

	operator vk::Device *() { return &m_Members->m_Device; }
	operator VkDevice *() { return reinterpret_cast<VkDevice *>(&m_Members->m_Device); }

	operator const vk::Device *() const { return &m_Members->m_Device; }
	operator const VkDevice *() const { return reinterpret_cast<const VkDevice *>(&m_Members->m_Device); }

	operator vk::PhysicalDevice *() const { return &m_Members->m_PhysicalDevice; }
	operator VkPhysicalDevice *() const { return reinterpret_cast<VkPhysicalDevice *>(&m_Members->m_PhysicalDevice); }

	operator const vk::PhysicalDevice *() const { return &m_Members->m_PhysicalDevice; }
	operator const VkPhysicalDevice *() const
	{
		return reinterpret_cast<const VkPhysicalDevice *>(&m_Members->m_PhysicalDevice);
	}

	vk::Device *operator->() { return &m_Members->m_Device; }

	bool valid() { return m_Members != nullptr; }
	void cleanup();

  private:
	struct DeviceMembers
	{
		DeviceMembers(vk::PhysicalDevice pDevice, vk::SurfaceKHR surface)
			: m_Indices(pDevice, surface)
		{
			m_PhysicalDevice = pDevice;
		}
		~DeviceMembers();

		QueueFamilyIndices m_Indices;
		vk::PhysicalDevice m_PhysicalDevice;
		vk::PhysicalDeviceMemoryProperties m_MemProps;
		vk::Device m_Device;
		vk::Queue m_GraphicsQueue;
		vk::Queue m_ComputeQueue;
		vk::Queue m_PresentQueue;
		vk::Queue m_TransferQueue;
		vk::CommandPool m_CommandPool;
	};

	std::shared_ptr<DeviceMembers> m_Members = nullptr;
};
} // namespace vkc
