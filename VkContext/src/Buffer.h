#pragma once

#include <vulkan/vulkan.hpp>

#include "Device.h"

namespace vkc
{
enum AllocLocation
{
	ON_DEVICE = 1,
	ON_HOST = 2
};

template <typename T> class Buffer
{
  public:
	Buffer(const Device &device, vk::DeviceSize elementCount, vk::MemoryPropertyFlags memFlags,
		   vk::BufferUsageFlags usageFlags, uint location = ON_DEVICE)
		: m_Device(device), m_Elements(elementCount), m_MemFlags(memFlags), m_UsageFlags(usageFlags), m_Flags(location)
	{
		vk::Device vkDevice = device.getVkDevice();

		vk::BufferCreateInfo createInfo{};
		createInfo.setPNext(nullptr);
		createInfo.setSize(elementCount * sizeof(T));
		createInfo.setUsage(usageFlags);
		createInfo.setSharingMode(vk::SharingMode::eExclusive);
		createInfo.setQueueFamilyIndexCount(0u);
		createInfo.setPQueueFamilyIndices(nullptr);

		m_Buffer = vkDevice.createBuffer(createInfo);
		const vk::MemoryRequirements memReqs = vkDevice.getBufferMemoryRequirements(m_Buffer);

		vk::MemoryAllocateInfo memoryAllocateInfo{};
		memoryAllocateInfo.setPNext(nullptr);
		memoryAllocateInfo.setAllocationSize(memReqs.size);
		memoryAllocateInfo.setMemoryTypeIndex(device.getMemoryType(memReqs, memFlags));

		m_Memory = vkDevice.allocateMemory(memoryAllocateInfo);

		vkDevice.bindBufferMemory(m_Buffer, m_Memory, 0);

		if (location & ON_HOST)
			m_HostBuffer = new T[elementCount];
	}
	~Buffer() { Cleanup(); }

	void Cleanup()
	{
		if (m_HostBuffer)
			delete[] m_HostBuffer;
		if (m_Buffer)
			m_Device->destroyBuffer(m_Buffer);
		if (m_Memory)
			m_Device->freeMemory(m_Memory);

		m_Flags = 0;
		m_HostBuffer = nullptr;
		m_Buffer = nullptr;
		m_Memory = nullptr;
		m_Elements = 0;
	}

	void CopyToDevice() { CopyToDevice(m_HostBuffer, GetSize()); }

	void CopyToDevice(const void *storage, vk::DeviceSize size = 0)
	{
		if (size == 0)
			size = m_Elements * sizeof(T);
		assert(size <= (m_Elements * sizeof(T)));

		if (CanMap())
		{
			void *memory = Map();
			memcpy(memory, storage, size);
			Unmap();
		}
		else
		{
			auto stagingBuffer = Buffer<uint8_t>(
				m_Device, size, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
				vk::BufferUsageFlagBits::eTransferSrc);
			memcpy(stagingBuffer.Map(), storage, size);
			stagingBuffer.Unmap();
			stagingBuffer.CopyTo(this);
		}
	}

	void CopyToHost()
	{
		assert(m_Flags & ON_HOST);
		assert(m_HostBuffer != nullptr);

		CopyToHost(m_HostBuffer);
	}

	void CopyToHost(void *storage)
	{
		if (CanMap())
		{
			void *memory = Map();
			memcpy(storage, memory, m_Elements * sizeof(T));
			Unmap();
		}
		else
		{
			assert(m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc);
			auto stagingBuffer =
				Buffer<uint8_t>(m_Device, GetSize(),
								vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
								vk::BufferUsageFlagBits::eTransferDst);
			this->CopyTo(&stagingBuffer);
			memcpy(storage, stagingBuffer.Map(), stagingBuffer.GetSize());
			stagingBuffer.Unmap();
		}
	}

	template <typename B> void CopyTo(Buffer<B> *buffer)
	{
		assert(m_UsageFlags & vk::BufferUsageFlagBits::eTransferSrc);
		assert(buffer->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);

		auto allocInfo = vk::CommandBufferAllocateInfo{};
		allocInfo.setPNext(nullptr);
		allocInfo.setCommandBufferCount(1);
		allocInfo.setCommandPool(m_Device.getCommandPool());
		allocInfo.setLevel(vk::CommandBufferLevel::ePrimary);
		vk::CommandBuffer cmdBuffer;

		m_Device->allocateCommandBuffers(&allocInfo, &cmdBuffer);
		cmdBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));

		vk::BufferCopy copyRegion = vk::BufferCopy(0, 0, m_Elements * sizeof(T));
		cmdBuffer.copyBuffer(m_Buffer, *buffer, 1, &copyRegion);
		cmdBuffer.end();

		// Submit to queue
		vk::SubmitInfo submitInfo{};
		submitInfo.setPNext(nullptr);
		submitInfo.setCommandBufferCount(1);
		submitInfo.setPCommandBuffers(&cmdBuffer);
		auto transferQueue = m_Device.getTransferQueue();
		transferQueue.submit(submitInfo, nullptr);

		// Wait until finished before freeing command buffer
		transferQueue.waitIdle();
		m_Device->freeCommandBuffers(m_Device.getCommandPool(), {cmdBuffer});
	}

	T *Map()
	{
		assert(CanMap());
		void *memory = m_Device->mapMemory(m_Memory, 0, m_Elements * sizeof(T));
		assert(memory);
		return (T *)(memory);
	}

	void Unmap()
	{
		assert(CanMap());
		m_Device->unmapMemory(m_Memory);
	}

	[[nodiscard]] vk::DescriptorBufferInfo GetDescriptorBufferInfo(vk::DeviceSize offset = 0,
																   vk::DeviceSize range = 0) const
	{
		vk::DescriptorBufferInfo info{};
		info.setBuffer(m_Buffer);
		info.setOffset(offset);
		info.setRange(range != 0 ? range : GetSize());
		return info;
	}

	operator vk::Buffer() const { return m_Buffer; }
	operator vk::DeviceMemory() const { return m_Memory; }
	operator vk::Buffer *() { return &m_Buffer; }
	operator vk::DeviceMemory *() { return &m_Memory; }
	operator const vk::Buffer *() const { return &m_Buffer; }
	operator const vk::DeviceMemory *() const { return &m_Memory; }
	operator vk::DescriptorBufferInfo() const { return GetDescriptorBufferInfo(0, 0); }

	constexpr bool CanMap() const
	{
		return (m_MemFlags & vk::MemoryPropertyFlagBits::eHostVisible) &&
			   (m_MemFlags & vk::MemoryPropertyFlagBits::eHostCoherent);
	}

	T *GetHostBuffer() { return m_HostBuffer; }
	[[nodiscard]] vk::DeviceSize GetElementCount() const { return m_Elements; }
	[[nodiscard]] vk::DeviceSize GetSize() const { return m_Elements * sizeof(T); }
	[[nodiscard]] vk::MemoryPropertyFlags GetMemoryProperties() const { return m_MemFlags; }
	[[nodiscard]] vk::BufferUsageFlags GetBufferUsageFlags() const { return m_UsageFlags; }

  private:
	uint m_Flags;
	T *m_HostBuffer = nullptr;
	Device m_Device;
	vk::Buffer m_Buffer = nullptr;
	vk::DeviceMemory m_Memory = nullptr;
	vk::DeviceSize m_Elements = 0;
	vk::MemoryPropertyFlags m_MemFlags;
	vk::BufferUsageFlags m_UsageFlags;
};

template <typename T, typename B>
static void RecordCopyCommand(Buffer<B> *target, Buffer<T> *source, vk::CommandBuffer &cmdBuffer)
{
	assert(target->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferDst);
	assert(source->GetBufferUsageFlags() & vk::BufferUsageFlagBits::eTransferSrc);

	const vk::DeviceSize copySize = std::min(target->GetSize(), source->GetSize());
	vk::BufferCopy copyRegion{};
	copyRegion.setSrcOffset(0);
	copyRegion.setDstOffset(0);
	copyRegion.setSize(copySize);
	cmdBuffer.copyBuffer(*source, *target, 1, &copyRegion);
}
} // namespace vkc