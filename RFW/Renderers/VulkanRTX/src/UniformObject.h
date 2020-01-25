#pragma once

#include <vulkan/vulkan.hpp>

#include <DeviceStructures.h>

#include "VulkanDevice.h"
#include "Buffer.h"
#include "VmaBuffer.h"

#if 1

namespace vkrtx
{
template <typename T> class UniformObject
{
  public:
	static_assert(sizeof(T) == 4 ||
				  (sizeof(T) % 8) == 0); // Make sure object is either at least 4 bytes big or 8 byte aligned
	UniformObject(const VulkanDevice &device, vk::BufferUsageFlagBits usage = vk::BufferUsageFlagBits(),
				  vk::DeviceSize count = 1)
		: m_Buffer(device)
	{
		if (count == 1)
			m_Data = new T;
		else
			m_Data = new T[count];

		m_Buffer.allocate(count, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
						  usage | vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst |
							  vk::BufferUsageFlagBits::eTransferSrc,
						  VMA_MEMORY_USAGE_CPU_TO_GPU);
	}
	~UniformObject() { cleanup(); }

	void copyToDevice() { m_Buffer.copyToDevice(m_Data, m_Buffer.getElementCount() * sizeof(T)); }
	void updateData(const T *data, uint32_t index = 0, uint32_t count = 1)
	{
		assert(index + count < m_Buffer.GetElementCount());
		memcpy(m_Data + index, data, count * sizeof(T));
	}

	T *get_data() { return m_Data; }

	T *readDataFromDevice()
	{
		m_Buffer.copyToHost(m_Data);
		return m_Data;
	}

	void cleanup()
	{
		if (m_Data)
		{
			if (m_Buffer.getElementCount() == 1)
				delete m_Data;
			else
				delete[] m_Data;
			m_Data = nullptr;
		}

		m_Buffer.cleanup();
	}

	[[nodiscard]] vk::Buffer getVkBuffer() const { return m_Buffer; }
	[[nodiscard]] vk::DeviceMemory getVkMemory() const { return m_Buffer; }
	VmaBuffer<T> &get_buffer() { return m_Buffer; }
	[[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo() const
	{
		return m_Buffer.getDescriptorBufferInfo();
	}

  private:
	T *m_Data;
	VmaBuffer<T> m_Buffer;
};
} // namespace vkrtx
#else

namespace vkrtx
{
template <typename T> class UniformObject
{
  public:
	static_assert(sizeof(T) == 4 ||
				  (sizeof(T) % 8) == 0); // Make sure object is either at least 4 bytes big or 8 byte aligned
	UniformObject(const VulkanDevice &device, vk::BufferUsageFlagBits usage = vk::BufferUsageFlagBits(),
				  vk::DeviceSize count = 1)
		: m_Buffer(device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
				   usage | vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst |
					   vk::BufferUsageFlagBits::eTransferSrc,
				   ON_HOST | ON_DEVICE)
	{
	}
	~UniformObject() { cleanup(); }

	void copyToDevice() { m_Buffer.copyToDevice(); }
	void updateData(const T *data, uint32_t index = 0, uint32_t count = 1)
	{
		assert(index + count < m_Buffer.GetElementCount());
		memcpy(m_Buffer.GetHostBuffer() + index, data, count * sizeof(T));
	}
	T *get_data() { return m_Buffer.GetHostBuffer(); }
	T *readDataFromDevice()
	{
		m_Buffer.copyToHost();
		return m_Buffer.GetHostBuffer();
	}

	void cleanup() { m_Buffer.cleanup(); }

	[[nodiscard]] vk::Buffer getVkBuffer() const { return m_Buffer.getVkBuffer(); }
	[[nodiscard]] vk::DeviceMemory getVkMemory() const { return m_Buffer.getVkMemory(); }
	Buffer<T> &get_buffer() { return m_Buffer; }
	[[nodiscard]] vk::DescriptorBufferInfo getDescriptorBufferInfo() const
	{
		return m_Buffer.getDescriptorBufferInfo();
	}

  private:
	Buffer<T> m_Buffer;
};
} // namespace vkrtx

#endif