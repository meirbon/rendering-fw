#pragma once

#include <vulkan/vulkan.hpp>

#include "VmaBuffer.h"

#include <memory>

namespace vkrtx
{
enum AllocLocation
{
	NOT_ALLOCATED = 0,
	ON_DEVICE = 1,
	ON_HOST = 2,
	ON_ALL = ON_DEVICE | ON_HOST
};

template <typename T> class Buffer
{
  private:
	struct Members
	{
		unsigned int m_Flags = NOT_ALLOCATED;
		T *m_HostBuffer = nullptr;
		VulkanDevice m_Device;
		vk::Buffer m_Buffer = nullptr;
		vk::DeviceMemory m_Memory = nullptr;
		vk::DeviceSize m_Elements = 0;
		vk::MemoryPropertyFlags m_MemFlags;
		vk::BufferUsageFlags m_UsageFlags;

		void cleanup()
		{
			if (m_Buffer)
			{
				if (m_Buffer)
					m_Device->destroyBuffer(m_Buffer);
				if (m_Memory)
					m_Device->freeMemory(m_Memory);
				m_Buffer = nullptr;
				m_Memory = nullptr;
				m_Elements = 0;
				m_Flags = 0;
			}

			if (m_HostBuffer)
			{
				delete[] m_HostBuffer;
				m_HostBuffer = nullptr;
			}
		}
	};
	std::shared_ptr<Members> m_Members;

  public:
	Buffer(const VulkanDevice &device)
	{
		m_Members = std::make_shared<Members>();
		m_Members->m_Device = device;
	}
	Buffer(const VulkanDevice &device, vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags,
		   vk::BufferUsageFlags usageFlags, unsigned int location = ON_DEVICE)
		: Buffer(device)
	{
		allocate(elementCount, memFlags, usageFlags, location);
	}

	~Buffer() { cleanup(); }

	void cleanup() { m_Members.reset(); }

	void reallocate(vk::DeviceSize elementCount, bool force = false)
	{
		if (elementCount < m_Members->m_Elements && !force)
			return;

		const auto flags = m_Members->m_Flags;
		const auto memFlags = m_Members->m_MemFlags;
		const auto usageFlags = m_Members->m_UsageFlags;
		const auto device = m_Members->m_Device;

		m_Members->cleanup();

		m_Members->m_Device = device;
		m_Members->m_Flags = flags;
		m_Members->m_MemFlags = memFlags;
		m_Members->m_UsageFlags = usageFlags;
		m_Members->m_Elements = elementCount;

		vk::Device vkDevice = m_Members->m_Device.getVkDevice();
		if (m_Members->m_Flags & ON_DEVICE)
		{
			auto createInfo = vk::BufferCreateInfo();
			createInfo.setPNext(nullptr);
			createInfo.setSize(elementCount * sizeof(T));
			createInfo.setUsage(usageFlags);
			createInfo.setSharingMode(vk::SharingMode::eExclusive);
			createInfo.setQueueFamilyIndexCount(0u);
			createInfo.setPQueueFamilyIndices(nullptr);

			m_Members->m_Buffer = vkDevice.createBuffer(createInfo);
			const vk::MemoryRequirements memReqs = vkDevice.getBufferMemoryRequirements(m_Members->m_Buffer);

			auto memoryAllocateInfo = vk::MemoryAllocateInfo();
			memoryAllocateInfo.setPNext(nullptr);
			memoryAllocateInfo.setAllocationSize(memReqs.size);
			memoryAllocateInfo.setMemoryTypeIndex(device.getMemoryType(memReqs, memFlags));

			m_Members->m_Memory = vkDevice.allocateMemory(memoryAllocateInfo);
			assert(m_Members->m_Memory);

			vkDevice.bindBufferMemory(m_Members->m_Buffer, m_Members->m_Memory, 0);
		}

		if (m_Members->m_Flags & ON_HOST)
		{
			m_Members->m_HostBuffer = new T[elementCount];
		}
	}

	void allocate(vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags, vk::BufferUsageFlags usageFlags,
				  unsigned int location = ON_DEVICE)
	{
		if (elementCount < m_Members->m_Elements)
			return;

		m_Members->cleanup();

		m_Members->m_Elements = elementCount;
		m_Members->m_UsageFlags = usageFlags;
		m_Members->m_MemFlags = memFlags;

		vk::Device vkDevice = m_Members->m_Device.getVkDevice();

		if (location & ON_DEVICE)
		{
			auto createInfo = vk::BufferCreateInfo();
			createInfo.setPNext(nullptr);
			createInfo.setSize(elementCount * sizeof(T));
			createInfo.setUsage(usageFlags);
			createInfo.setSharingMode(vk::SharingMode::eExclusive);
			createInfo.setQueueFamilyIndexCount(0u);
			createInfo.setPQueueFamilyIndices(nullptr);

			m_Members->m_Buffer = vkDevice.createBuffer(createInfo);
			const vk::MemoryRequirements memReqs = vkDevice.getBufferMemoryRequirements(m_Members->m_Buffer);

			auto memoryAllocateInfo = vk::MemoryAllocateInfo();
			memoryAllocateInfo.setPNext(nullptr);
			memoryAllocateInfo.setAllocationSize(memReqs.size);
			memoryAllocateInfo.setMemoryTypeIndex(m_Members->m_Device.getMemoryType(memReqs, memFlags));

			m_Members->m_Memory = vkDevice.allocateMemory(memoryAllocateInfo);
			assert(m_Members->m_Memory);

			vkDevice.bindBufferMemory(m_Members->m_Buffer, m_Members->m_Memory, 0);
		}

		if (location & ON_HOST)
		{
			m_Members->m_HostBuffer = new T[elementCount];
		}
	}

	void copyToDevice() { copyToDevice(m_Members->m_HostBuffer, getSize()); }

	void copyToDeviceAsync() { copyToDeviceAsync(m_Members->m_HostBuffer, getSize()); }

	void copyToDevice(const T &data) { copyToDevice(&data, 0); }

	void copyToDevice(const void *storage, vk::DeviceSize size = 0)
	{
		assert(size <= (m_Members->m_Elements * sizeof(T)));
		if (size == 0)
			size = m_Members->m_Elements * sizeof(T);

		if (canMap())
		{
			void *memory = map();
			memcpy(memory, storage, size);
			unmap();
		}
		else
		{
			auto stagingBuffer =
				Buffer<uint8_t>(m_Members->m_Device, size,
								vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
								vk::BufferUsageFlagBits::eTransferSrc);
			memcpy(stagingBuffer.map(), storage, size);
			stagingBuffer.unmap();
			stagingBuffer.copyTo(this);
			stagingBuffer.cleanup();
		}
	}

	void copyToHost()
	{
		assert(m_Members->m_Flags & ON_HOST);
		assert(m_Members->m_HostBuffer != nullptr);

		copyToHost(m_Members->m_HostBuffer);
	}

	void copyToHost(void *storage)
	{
		if (canMap())
		{
			void *memory = map();
			memcpy(storage, memory, m_Members->m_Elements * sizeof(T));
			unmap();
		}
		else
		{
			assert(m_Members->m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc);
			auto stagingBuffer =
				Buffer<uint8_t>(m_Members->m_Device, getSize(),
								vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
								vk::BufferUsageFlagBits::eTransferDst);
			copyTo(&stagingBuffer);
			memcpy(storage, stagingBuffer.map(), stagingBuffer.getSize());
			stagingBuffer.unmap();
			stagingBuffer.cleanup();
		}
	}

	template <typename B> void copyTo(Buffer<B> *buffer)
	{
		assert(m_Members->m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc);
		assert(buffer->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);

		auto cmdBuffer =
			m_Members->m_Device.createOneTimeCmdBuffer(vk::CommandBufferLevel::ePrimary, VulkanDevice::TRANSFER);
		vk::BufferCopy copyRegion = vk::BufferCopy(0, 0, m_Members->m_Elements * sizeof(T));
		cmdBuffer->copyBuffer(m_Members->m_Buffer, *buffer, 1, &copyRegion);

		auto transferQueue = m_Members->m_Device.getTransferQueue();
		cmdBuffer.submit(transferQueue, true);
	}

	template <typename B> void copyTo(VmaBuffer<B> *buffer)
	{
		assert(m_Members->m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc);
		assert(buffer->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);

		auto cmdBuffer =
			m_Members->m_Device.createOneTimeCmdBuffer(vk::CommandBufferLevel::ePrimary, VulkanDevice::TRANSFER);
		vk::BufferCopy copyRegion = vk::BufferCopy(0, 0, m_Members->m_Elements * sizeof(T));
		cmdBuffer->copyBuffer(m_Members->m_Buffer, *buffer, 1, &copyRegion);

		auto transferQueue = m_Members->m_Device.getTransferQueue();
		cmdBuffer.submit(transferQueue, true);
	}

	T *map()
	{
		assert(canMap());
		void *memory = m_Members->m_Device->mapMemory(m_Members->m_Memory, 0, m_Members->m_Elements * sizeof(T));
		assert(memory);
		return (T *)(memory);
	}

	void unmap()
	{
		assert(canMap());
		m_Members->m_Device->unmapMemory(m_Members->m_Memory);
	}

	[[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo(vk::DeviceSize offset = 0,
																   vk::DeviceSize range = 0) const
	{
		vk::DescriptorBufferInfo info{};
		info.setBuffer(m_Members->m_Buffer);
		info.setOffset(offset);
		info.setRange(range != 0 ? range : getSize());
		return info;
	}

	operator vk::Buffer() const { return m_Members->m_Buffer; }
	operator vk::DeviceMemory() const { return m_Members->m_Memory; }
	operator vk::Buffer *() { return &m_Members->m_Buffer; }
	operator vk::DeviceMemory *() { return &m_Members->m_Memory; }
	operator const vk::Buffer *() const { return &m_Members->m_Buffer; }
	operator const vk::DeviceMemory *() const { return &m_Members->m_Memory; }
	operator vk::DescriptorBufferInfo() const { return getDescriptorBufferInfo(0, 0); }

	[[nodiscard]] constexpr bool canMap() const
	{
		return (m_Members->m_MemFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
			   (m_Members->m_MemFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
	}

	T *GetHostBuffer() { return m_Members->m_HostBuffer; }
	[[nodiscard]] vk::DeviceMemory getDeviceMemory() const { return m_Members->m_Memory; }
	[[nodiscard]] vk::DeviceSize getElementCount() const { return m_Members->m_Elements; }
	[[nodiscard]] vk::DeviceSize getSize() const { return m_Members->m_Elements * sizeof(T); }
	[[nodiscard]] vk::MemoryPropertyFlags getMemoryProperties() const { return m_Members->m_MemFlags; }
	[[nodiscard]] vk::BufferUsageFlags getBufferUsageFlags() const { return m_Members->m_UsageFlags; }
};

template <typename T, typename B>
static void recordCopyCommand(Buffer<B> *target, Buffer<T> *source, vk::CommandBuffer &cmdBuffer)
{
	assert(target->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);
	assert(source->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc);

	const vk::DeviceSize copySize = std::min(target->getSize(), source->getSize());
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset(0);
	copyRegion.setDstOffset(0);
	copyRegion.setSize(copySize);
	cmdBuffer.copyBuffer(*source, *target, 1, &copyRegion);
}

template <typename T, typename B>
static void recordCopyCommand(Buffer<B> *target, VmaBuffer<T> *source, vk::CommandBuffer &cmdBuffer)
{
	assert(target->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);
	assert(source->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc);

	const vk::DeviceSize copySize = std::min(target->getSize(), source->getSize());
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset(0);
	copyRegion.setDstOffset(0);
	copyRegion.setSize(copySize);
	cmdBuffer.copyBuffer(*source, *target, 1, &copyRegion);
}

template <typename T, typename B>
static void recordCopyCommand(VmaBuffer<B> *target, Buffer<T> *source, vk::CommandBuffer &cmdBuffer)
{
	assert(target->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);
	assert(source->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc);

	const vk::DeviceSize copySize = std::min(target->getSize(), source->getSize());
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset(0);
	copyRegion.setDstOffset(0);
	copyRegion.setSize(copySize);
	cmdBuffer.copyBuffer(*source, *target, 1, &copyRegion);
}

template <typename T, typename B>
static void recordCopyCommand(VmaBuffer<B> *target, VmaBuffer<T> *source, vk::CommandBuffer &cmdBuffer)
{
	assert(target->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);
	assert(source->getBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc);

	const vk::DeviceSize copySize = std::min(target->getSize(), source->getSize());
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset(0);
	copyRegion.setDstOffset(0);
	copyRegion.setSize(copySize);
	cmdBuffer.copyBuffer(*source, *target, 1, &copyRegion);
}
} // namespace vkrtx
