//
// Created by Mèir Noordermeer on 09/10/2019.
//

#ifndef RENDERINGFW_OPTIX6CONTEXT_SRC_OPTIXCUDABUFFER_H
#define RENDERINGFW_OPTIX6CONTEXT_SRC_OPTIXCUDABUFFER_H

#include <optix.h>
#include <optix_world.h>
#include <vector>
#include <MathIncludes.h>
#include <array>

enum OptiXBufferType
{
	Read = RT_BUFFER_INPUT,
	ReadWrite = RT_BUFFER_INPUT_OUTPUT
};

template <typename T> class OptiXCUDABuffer
{
  public:
	OptiXCUDABuffer() = default;
	OptiXCUDABuffer(optix::Context context, const std::vector<T> &data, OptiXBufferType bufferType, RTformat rtFormat)
		: m_Context(context), m_Dimensions(1)
	{
		m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat, data.size());
		if (rtFormat == RT_FORMAT_USER)
			m_Buffer->setElementSize(sizeof(T));
		if (bufferType == Read)
		{
			cudaMalloc(&m_DevicePointer, data.size() * sizeof(T));
			m_CUDAAllocated = true;
			m_Buffer->setDevicePointer(0 /*Not considering multi gpu setups*/, m_DevicePointer);
			cudaMemcpy(m_DevicePointer, data.data(), data.size() * sizeof(T), cudaMemcpyHostToDevice);
		}
		else
		{
			m_Buffer->setSize(data.size());
			m_DevicePointer = reinterpret_cast<T *>(m_Buffer->getDevicePointer(0 /*Not considering multi gpu setups*/));
			memcpy(m_Buffer->map(), data.data(), data.size() * sizeof(T));
			m_Buffer->unmap();
		}

		assert(m_DevicePointer);
	}

	OptiXCUDABuffer(optix::Context context, size_t elementCount, OptiXBufferType bufferType, RTformat rtFormat)
		: m_Context(context), m_Width(elementCount), m_Dimensions(1)
	{
		if (bufferType == Read)
		{
			m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat);
			if (rtFormat == RT_FORMAT_USER)
				m_Buffer->setElementSize(sizeof(T));
			cudaMalloc(&m_DevicePointer, elementCount * sizeof(T));
			m_CUDAAllocated = true;
			m_Buffer->setDevicePointer(0 /*Not considering multi gpu setups*/, m_DevicePointer);
		}
		else
		{
			m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat, elementCount);
			if (rtFormat == RT_FORMAT_USER)
				m_Buffer->setElementSize(sizeof(T));
			m_DevicePointer = (T *)m_Buffer->getDevicePointer(0 /*Not considering multi gpu setups*/);
		}

		if (rtFormat == RT_FORMAT_USER)
			m_Buffer->setElementSize(sizeof(T));
		assert(m_DevicePointer);
	}

	template <size_t B>
	OptiXCUDABuffer(optix::Context context, const std::array<size_t, B> &dimensions, OptiXBufferType bufferType,
					RTformat rtFormat)
		: m_Context(context), m_Dimensions(dimensions.size())
	{
		static_assert(B <= 2 && B >= 1);
		switch (B)
		{
		case (1):
			m_Width = dimensions.at(0);
			if (bufferType == Read)
			{
				m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat);
				if (rtFormat == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				CheckCUDA(cudaMalloc(&m_DevicePointer, m_Width * sizeof(T)));
				m_CUDAAllocated = true;
				m_Buffer->setDevicePointer(0 /*Not considering multi gpu setups*/, m_DevicePointer);
			}
			else
			{
				m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat, m_Width);
				if (rtFormat == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				m_DevicePointer = (T *)m_Buffer->getDevicePointer(0 /*Not considering multi gpu setups*/);
			}

			break;
		case (2):
			m_Width = dimensions.at(0);
			m_Height = dimensions.at(1);
			if (bufferType == Read)
			{
				m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat);
				if (rtFormat == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				CheckCUDA(cudaMalloc(&m_DevicePointer, m_Width * m_Height * sizeof(T)));
				m_CUDAAllocated = true;
				m_Buffer->setDevicePointer(0 /*Not considering multi gpu setups*/, m_DevicePointer);
			}
			else
			{
				m_Buffer = context->createBufferForCUDA((uint)bufferType, rtFormat, m_Width, m_Height);
				if (rtFormat == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				m_DevicePointer = (T *)m_Buffer->getDevicePointer(0 /*Not considering multi gpu setups*/);
			}
			break;
		}
		assert(m_DevicePointer);
	}

	~OptiXCUDABuffer() { cleanup(); }

	void cleanup()
	{
		if (m_Buffer.get())
			m_Buffer->destroy();
		m_Buffer = nullptr;
		if (m_CUDAAllocated && m_DevicePointer)
			cudaFree(m_DevicePointer);
		m_DevicePointer = nullptr;
	}

	T *map() { return (T *)m_Buffer->map(); }
	void unmap() { m_Buffer->unmap(); }
	T *getDevicePointer() { return m_DevicePointer; }

	template <size_t B>
	void copyToDevice(const T *data, const std::array<size_t, B> &dimensions, size_t offset = 0, bool async = false)
	{
		assert(m_CUDAAllocated);
		assert(dimensions.size() <= 2 && dimensions.size() >= 1);
		switch (dimensions.size())
		{
		case (1):
			if (async)
			{
				CheckCUDA(cudaMemcpyAsync(m_DevicePointer + offset, data, dimensions.at(0) * sizeof(T),
										  cudaMemcpyHostToDevice));
			}
			else
			{
				CheckCUDA(
					cudaMemcpy(m_DevicePointer + offset, data, dimensions.at(0) * sizeof(T), cudaMemcpyHostToDevice));
			}
			break;
		case (2):
			if (async)
			{

				CheckCUDA(cudaMemcpy2DAsync(m_DevicePointer + offset, m_Height, data, m_Height, dimensions.at(0),
											dimensions.at(1), cudaMemcpyHostToDevice));
			}
			else
			{
				CheckCUDA(cudaMemcpy(m_DevicePointer + offset, m_Height, data, m_Height, dimensions.at(0),
									 dimensions.at(1), cudaMemcpyHostToDevice));
			}
			break;
		}
	}

	optix::Buffer getOptiXBuffer() const { return m_Buffer; }

	void clear()
	{
		assert(!m_CUDAAllocated);
		switch (m_Dimensions)
		{
		case (1):
			CheckCUDA(cudaMemset(m_DevicePointer, 0, m_Width * m_Height * sizeof(T)));
			break;
		case (2):
			CheckCUDA(cudaMemset2D(m_DevicePointer, m_Width * sizeof(T), 0, m_Width, m_Height));
			break;
		}
	}

	void clearAsync()
	{
		assert(!m_CUDAAllocated);
		switch (m_Dimensions)
		{
		case (1):
			CheckCUDA(cudaMemsetAsync(m_DevicePointer, 0, m_Width * m_Height * sizeof(T)));
			break;
		case (2):
			CheckCUDA(cudaMemset2DAsync(m_DevicePointer, m_Width * sizeof(T), 0, m_Width, m_Height));
			break;
		}
	}

  private:
	bool m_CUDAAllocated = false;
	optix::Context m_Context;
	optix::Buffer m_Buffer;
	size_t m_Dimensions = 0;
	size_t m_Width = 0;
	size_t m_Height = 0;
	T *m_DevicePointer = nullptr;
};

#endif // RENDERINGFW_OPTIX6CONTEXT_SRC_OPTIXCUDABUFFER_H
