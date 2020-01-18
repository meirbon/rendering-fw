#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <cassert>
#include <cstdio>
#include <vector>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <MathIncludes.h>
#include "CheckCUDA.h"

enum BufferLocation
{
	ON_HOST = 2u,
	ON_DEVICE = 4u,
	ON_ALL = 2u | 4u
};

template <typename T> class CUDABuffer
{
  public:
	CUDABuffer() = default;
	CUDABuffer(const std::vector<T> &data, uint flags = ON_ALL, bool async = false) : CUDABuffer(data.size(), flags)
	{
		if (flags & ON_HOST)
			m_HostData = data;
		if (flags & ON_DEVICE)
		{
			if (async)
			{
				CheckCUDA(cudaMemcpyAsync(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
			}
			else
			{
				CheckCUDA(cudaMemcpy(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
			}
		}
	}

	CUDABuffer(size_t elementCount, uint flags = ON_ALL) : m_Elements(static_cast<uint>(elementCount)), m_Flags(flags)
	{
		const bool allocateHost = (flags & ON_HOST);
		const bool allocateDevice = (flags & ON_DEVICE);

		if (allocateHost)
			m_HostData.resize(elementCount);

		if (allocateDevice)
			CheckCUDA(cudaMalloc(&m_DevicePointer, elementCount * sizeof(T)));
	}

	CUDABuffer(const T *data, size_t elementCount, uint flags = ON_ALL) : m_Elements(static_cast<uint>(elementCount)), m_Flags(flags)
	{
		const bool allocateHost = (flags & ON_HOST);
		const bool allocateDevice = (flags & ON_DEVICE);

		if (allocateHost)
			m_HostData = std::vector<T>(data, data + elementCount);

		if (allocateDevice)
		{
			CheckCUDA(cudaMalloc(&m_DevicePointer, elementCount * sizeof(T)));
			copy_to_device(data, elementCount, 0);
		}
	}
	~CUDABuffer() { cleanup(); }

	void cleanup()
	{
		m_HostData.clear();
		if (m_DevicePointer != nullptr)
			CheckCUDA(cudaFree(m_DevicePointer));

		m_DevicePointer = nullptr;
		m_Flags = 0;
	}

	T *data() { return m_HostData.data(); }
	T *device_data() { return m_DevicePointer; }

	const T *data() const { return m_HostData.data(); }
	const T *device_data() const { return m_DevicePointer; }

	void copy_to_host() { CheckCUDA(cudaMemcpy(m_HostData.data(), m_DevicePointer, m_Elements * sizeof(T), cudaMemcpyDeviceToHost)); }
	void copy_to_host_async() { CheckCUDA(cudaMemcpyAsync(m_HostData.data(), m_DevicePointer, m_Elements * sizeof(T), cudaMemcpyDeviceToHost)); }
	void copy_to_device(const T *data, size_t elementCount, size_t offset = 0)
	{
		assert(data);
		assert((elementCount + offset) <= m_Elements);
		CheckCUDA(cudaMemcpy(m_DevicePointer + offset, data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
	}

	void copy_to_device(const std::vector<T> &data, size_t offset = 0)
	{
		assert((m_Elements - offset) >= data.size());
		copy_to_device(data.data(), data.size(), offset);
	}

	void copy_to_device_async(const T *data, size_t elementCount, size_t offset = 0)
	{
		assert(data);
		assert((elementCount + offset) <= m_Elements);
		CheckCUDA(cudaMemcpyAsync(m_DevicePointer + offset, data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copy_to_device_async(const std::vector<T> &data, size_t offset = 0)
	{
		assert((data.size() + offset) <= m_Elements);
		CheckCUDA(cudaMemcpyAsync(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copy_to_device()
	{
		assert(m_Flags & ON_HOST);
		CheckCUDA(cudaMemcpy(m_DevicePointer, m_HostData.data(), m_Elements * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copy_to_device_async()
	{
		assert(m_Flags & ON_HOST);
		CheckCUDA(cudaMemcpyAsync(m_DevicePointer, m_HostData.data(), m_Elements * sizeof(T), cudaMemcpyHostToDevice));
	}
	void clear()
	{
		if (m_Flags & ON_DEVICE)
			CheckCUDA(cudaMemset(m_DevicePointer, 0, m_Elements * sizeof(T)));
		if (m_Flags & ON_HOST)
			memset(m_HostData.data(), 0, m_Elements * sizeof(T));
	}
	void clear_async()
	{
		if (m_Flags & ON_DEVICE)
			CheckCUDA(cudaMemsetAsync(m_DevicePointer, 0, m_Elements * sizeof(T)));
		if (m_Flags & ON_HOST)
			memset(m_HostData.data(), 0, m_Elements * sizeof(T));
	}

	uint size() const { return m_Elements; }

  private:
	uint m_Elements = 0;
	uint m_Flags = 0;
	std::vector<T> m_HostData;
	T *m_DevicePointer = nullptr;
};