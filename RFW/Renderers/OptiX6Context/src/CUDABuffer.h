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
				CheckCUDA(
					cudaMemcpyAsync(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
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

	CUDABuffer(const T *data, size_t elementCount, uint flags = ON_ALL)
		: m_Elements(static_cast<uint>(elementCount)), m_Flags(flags)
	{
		const bool allocateHost = (flags & ON_HOST);
		const bool allocateDevice = (flags & ON_DEVICE);

		if (allocateHost)
			m_HostData = std::vector<T>(data, data + elementCount);

		if (allocateDevice)
		{
			CheckCUDA(cudaMalloc(&m_DevicePointer, elementCount * sizeof(T)));
			copyToDevice(data, elementCount, 0);
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

	T *getHostPointer() { return m_HostData.data(); }
	T *getDevicePointer() { return m_DevicePointer; }

	void copyToHost()
	{
		CheckCUDA(cudaMemcpy(m_HostData.data(), m_DevicePointer, m_Elements * sizeof(T), cudaMemcpyDeviceToHost));
	}
	void copyToHostAsync()
	{
		CheckCUDA(cudaMemcpyAsync(m_HostData.data(), m_DevicePointer, m_Elements * sizeof(T), cudaMemcpyDeviceToHost));
	}
	void copyToDevice(const T *data, size_t elementCount, size_t offset = 0)
	{
		assert(data);
		assert((elementCount + offset) <= m_Elements);
		CheckCUDA(cudaMemcpy(m_DevicePointer + offset, data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copyToDevice(const std::vector<T> &data, size_t offset = 0)
	{
		assert((m_Elements - offset) >= data.size());
		copyToDevice(data.data(), data.size(), offset);
	}
	void copyToDeviceAsync(const T *data, size_t elementCount, size_t offset = 0)
	{
		assert(data);
		assert((elementCount + offset) <= m_Elements);
		CheckCUDA(cudaMemcpyAsync(m_DevicePointer + offset, data, elementCount * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copyToDeviceAsync(const std::vector<T> &data, size_t offset = 0)
	{
		assert((data.size() + offset) <= m_Elements);
		CheckCUDA(cudaMemcpyAsync(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copyToDevice()
	{
		assert(m_Flags & ON_HOST);
		CheckCUDA(cudaMemcpy(m_DevicePointer, m_HostData.data(), m_Elements * sizeof(T), cudaMemcpyHostToDevice));
	}
	void copyToDeviceAsync()
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
	void clearAsync()
	{
		if (m_Flags & ON_DEVICE)
			CheckCUDA(cudaMemsetAsync(m_DevicePointer, 0, m_Elements * sizeof(T)));
		if (m_Flags & ON_HOST)
			memset(m_HostData.data(), 0, m_Elements * sizeof(T));
	}

	uint getElementCount() const { return m_Elements; }

  private:
	uint m_Elements{};
	uint m_Flags = 0;
	std::vector<T> m_HostData;
	T *m_DevicePointer = nullptr;
};