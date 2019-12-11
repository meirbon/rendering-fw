#pragma once

#include <vulkan/vulkan.hpp>

#include <vk_mem_alloc.h>
#include <utils/Logger.h>

#include "CheckVK.h"
#include "VulkanDevice.h"

namespace vkrtx
{

template <typename T> class Buffer;
template <typename T> class VmaBuffer
{
  private:
	struct Members
	{
		Members() = default;
		~Members() { cleanup(); }

		void cleanup()
		{
			if (allocation)
			{
				vmaDestroyBuffer(allocator, (VkBuffer)buffer, allocation);
				allocation = nullptr;
				buffer = nullptr;
				elements = 0;
			}
		}

		VmaAllocationInfo allocInfo = {};
		VmaAllocator allocator = nullptr;
		VmaAllocation allocation = nullptr;
		VmaPool pool = nullptr;
		VmaMemoryUsage usage;

		VulkanDevice device;
		vk::Buffer buffer = nullptr;
		vk::DeviceSize elements = 0;

		vk::MemoryPropertyFlags memFlags;
		vk::BufferUsageFlags usageFlags;
	};

  public:
	VmaBuffer(const VulkanDevice &device, VmaPool pool = nullptr)
	{
		m_Members = std::make_shared<Members>();
		m_Members->device = device;
		m_Members->allocator = device.getAllocator();
		m_Members->pool = pool;

		assert(m_Members->device);
	}

	VmaBuffer(const VulkanDevice &device, vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags,
			  vk::BufferUsageFlags usageFlags, VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY,
			  bool forceFlags = false, VmaPool pool = nullptr)
	{
		m_Members = std::make_shared<Members>();
		m_Members->device = device;
		m_Members->allocator = device.getAllocator();
		m_Members->pool = pool;
		allocate(elementCount, memFlags, usageFlags, usage, forceFlags);

		assert(m_Members->device);
	}

	~VmaBuffer() { cleanup(); }

	void cleanup() { m_Members->cleanup(); }

	/*
	 * Reallocate a buffer with same settings but different size
	 */
	void reallocate(vk::DeviceSize elementCount, bool force = false)
	{
		assert(m_Members->device);

		if (elementCount < m_Members->elements && !force)
			return;

		const auto memFlags = m_Members->memFlags;
		const auto usageFlags = m_Members->usageFlags;
		const auto usage = m_Members->usage;
		auto allocator = m_Members->allocator;

		cleanup();
		m_Members = std::make_shared<Members>();

		m_Members->allocator = allocator;
		m_Members->memFlags = memFlags;
		m_Members->usageFlags = usageFlags;
		m_Members->elements = elementCount;

		auto createInfo = vk::BufferCreateInfo();
		createInfo.setPNext(nullptr);
		createInfo.setSize(elementCount * sizeof(T));
		createInfo.setUsage(usageFlags);
		createInfo.setSharingMode(vk::SharingMode::eExclusive);
		createInfo.setQueueFamilyIndexCount(0u);
		createInfo.setPQueueFamilyIndices(nullptr);

		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = usage;
		allocInfo.pool = m_Members->pool;
		CheckVK(vmaCreateBuffer(m_Members->allocator, (VkBufferCreateInfo *)&createInfo, &allocInfo,
								(VkBuffer *)&m_Members->buffer, &m_Members->allocation, &m_Members->allocInfo));
		assert(m_Members->allocInfo.deviceMemory);
	}

	/*
	 * (Re-)initialize a buffer with specified settings
	 */
	void allocate(vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags, vk::BufferUsageFlags usageFlags,
				  VmaMemoryUsage usage = VMA_MEMORY_USAGE_GPU_ONLY, bool forceFlags = false,
				  VkMemoryRequirements memReqs = {})
	{
		assert(m_Members);
		assert(m_Members->device);
		assert(m_Members->allocator);
		cleanup();

		if (elementCount < m_Members->elements)
			return;

		m_Members->elements = elementCount;
		m_Members->usageFlags = usageFlags;
		m_Members->memFlags = memFlags;
		m_Members->usage = usage;

		auto createInfo = vk::BufferCreateInfo();
		createInfo.setPNext(nullptr);
		createInfo.setSize(elementCount * sizeof(T));
		createInfo.setUsage(usageFlags);
		createInfo.setSharingMode(vk::SharingMode::eExclusive);
		createInfo.setQueueFamilyIndexCount(0u);
		createInfo.setPQueueFamilyIndices(nullptr);

		VmaAllocationCreateInfo allocInfo = {};
		allocInfo.usage = usage;
		allocInfo.memoryTypeBits = memReqs.memoryTypeBits;
		allocInfo.pool = m_Members->pool;

		if (forceFlags)
			allocInfo.requiredFlags = (VkMemoryPropertyFlags)m_Members->memFlags;
		else
			allocInfo.preferredFlags = (VkMemoryPropertyFlags)m_Members->memFlags;

		CheckVK(vmaCreateBuffer(m_Members->allocator, (VkBufferCreateInfo *)&createInfo, &allocInfo,
								(VkBuffer *)&m_Members->buffer, &m_Members->allocation, &m_Members->allocInfo));
		assert(m_Members->allocInfo.deviceMemory);
	}

	void copyToDevice(const void *storage, vk::DeviceSize size = 0)
	{
		assert(size <= (m_Members->elements * sizeof(T)));
		assert(m_Members->device);

		if (size == 0)
			size = m_Members->elements * sizeof(T);

		if (canMap())
		{
			void *memory = map();
			memcpy(memory, storage, size);
			unmap();
		}
		else
		{
			auto stagingBuffer =
				VmaBuffer<uint8_t>(m_Members->device, size, vk::MemoryPropertyFlagBits::eHostVisible,
								   vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_TO_GPU, true);
			memcpy(stagingBuffer.map(), storage, size);
			stagingBuffer.unmap();
			stagingBuffer.copyTo(this);
			stagingBuffer.cleanup();
		}
	}

	void copyToHost(void *storage)
	{
		assert(m_Members->device);
		if (canMap())
		{
			void *memory = map();
			memcpy(storage, memory, m_Members->elements * sizeof(T));
			unmap();
		}
		else
		{
			assert(m_Members->usageFlags & vk::BufferUsageFlagBits::eTransferSrc);
			auto stagingBuffer =
				VmaBuffer<uint8_t>(m_Members->device, getSize(), vk::MemoryPropertyFlagBits::eHostVisible,
								   vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_TO_CPU, true);
			this->copyTo(&stagingBuffer);
			memcpy(storage, stagingBuffer.map(), stagingBuffer.getSize());
			stagingBuffer.unmap();
		}
	}

	template <typename B> void copyTo(VmaBuffer<B> *buffer)
	{
		assert(m_Members->device);
		assert(m_Members->usageFlags & vk::BufferUsageFlagBits::eTransferSrc);
		assert(buffer->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);

		auto cmdBuffer =
			m_Members->device.createOneTimeCmdBuffer(vk::CommandBufferLevel::ePrimary, VulkanDevice::TRANSFER);
		vk::BufferCopy copyRegion = vk::BufferCopy(0, 0, m_Members->elements * sizeof(T));
		cmdBuffer->copyBuffer(m_Members->buffer, *buffer, 1, &copyRegion);

		auto transferQueue = m_Members->device.getTransferQueue();
		cmdBuffer.submit(transferQueue, true);
	}

	template <typename B> void copyTo(Buffer<B> *buffer)
	{
		assert(m_Members->device);
		assert(m_Members->usageFlags & vk::BufferUsageFlagBits::eTransferSrc);
		assert(buffer->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);

		auto cmdBuffer =
			m_Members->device.createOneTimeCmdBuffer(vk::CommandBufferLevel::ePrimary, VulkanDevice::TRANSFER);
		vk::BufferCopy copyRegion = vk::BufferCopy(0, 0, m_Members->elements * sizeof(T));
		cmdBuffer->copyBuffer(*this, *buffer, 1, &copyRegion);

		auto transferQueue = m_Members->device.getTransferQueue();
		cmdBuffer.submit(transferQueue, true);
	}

	T *map()
	{
		assert(m_Members->device);
		if (!canMap())
			throw std::runtime_error("Memory not mappable.");

		void *memory = nullptr;
		CheckVK(vmaMapMemory(m_Members->allocator, m_Members->allocation, &memory));
		assert(memory);
		return (T *)(memory);
	}

	void unmap()
	{
		assert(m_Members->device);
		assert(canMap());
		vmaUnmapMemory(m_Members->allocator, m_Members->allocation);
	}

	[[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo(vk::DeviceSize offset = 0,
																   vk::DeviceSize range = 0) const
	{
		assert(m_Members->device);
		vk::DescriptorBufferInfo info{};
		info.setBuffer(m_Members->buffer);
		info.setOffset(offset);
		info.setRange(range != 0 ? range : getSize());
		return info;
	}

	vk::DeviceSize getElementCount() const { return m_Members->elements; }
	vk::DeviceSize getSize() const { return m_Members->elements * sizeof(T); }
	vk::DeviceSize getOffset() const { return m_Members->allocInfo.offset; }
	vk::DeviceMemory getDeviceMemory() const { return m_Members->allocInfo.deviceMemory; };
	[[nodiscard]] vk::MemoryPropertyFlags getMemoryProperties() const { return m_Members->memFlags; }
	[[nodiscard]] vk::BufferUsageFlags getBufferUsageFlags() const { return m_Members->usageFlags; }

	operator vk::Buffer() const { return m_Members->buffer; }
	operator vk::DeviceMemory() const { return getDeviceMemory(); }
	operator vk::Buffer *() { return &m_Members->buffer; }
	operator vk::DescriptorBufferInfo() const { return getDescriptorBufferInfo(0, 0); }
	operator bool() const { return m_Members->elements > 0; }

	[[nodiscard]] bool canMap() const
	{
		VkMemoryPropertyFlags memFlags;
		vmaGetMemoryTypeProperties(m_Members->allocator, m_Members->allocInfo.memoryType, &memFlags);
		return (memFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
	}

  private:
	std::shared_ptr<Members> m_Members;
};
} // namespace vkrtx
