#pragma once

#include <optix.h>
#include <optix_world.h>
#include <vector>
#include <MathIncludes.h>
#include <array>

#include <utils/ArrayProxy.h>

enum OptiXBufferType
{
	Read = RT_BUFFER_INPUT,
	ReadWrite = RT_BUFFER_INPUT_OUTPUT
};

template <typename T, OptiXBufferType bufferType> class OptiXCUDABuffer
{
  public:
	static RTformat parse_format()
	{
		auto rtFormat = RT_FORMAT_USER;
		if (std::is_same<float, T>::value)
			rtFormat = RT_FORMAT_FLOAT;
		else if (std::is_same<glm::vec2, T>::value)
			rtFormat = RT_FORMAT_FLOAT2;
		else if (std::is_same<glm::vec3, T>::value)
			rtFormat = RT_FORMAT_FLOAT3;
		else if (std::is_same<glm::vec4, T>::value)
			rtFormat = RT_FORMAT_FLOAT4;
		else if (std::is_same<unsigned int, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_INT;
		else if (std::is_same<glm::uint, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_INT;
		else if (std::is_same<glm::uvec2, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_INT2;
		else if (std::is_same<glm::uvec3, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_INT3;
		else if (std::is_same<glm::uvec4, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_INT4;
		else if (std::is_same<int, T>::value)
			rtFormat = RT_FORMAT_INT;
		else if (std::is_same<glm::ivec2, T>::value)
			rtFormat = RT_FORMAT_INT2;
		else if (std::is_same<glm::ivec3, T>::value)
			rtFormat = RT_FORMAT_INT3;
		else if (std::is_same<glm::ivec4, T>::value)
			rtFormat = RT_FORMAT_INT4;
		else if (std::is_same<short, T>::value)
			rtFormat = RT_FORMAT_SHORT;
		else if (std::is_same<glm::i16vec2, T>::value)
			rtFormat = RT_FORMAT_SHORT2;
		else if (std::is_same<glm::i16vec3, T>::value)
			rtFormat = RT_FORMAT_SHORT3;
		else if (std::is_same<glm::i16vec4, T>::value)
			rtFormat = RT_FORMAT_SHORT4;
		else if (std::is_same<unsigned short, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_SHORT;
		else if (std::is_same<glm::u16vec2, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_SHORT2;
		else if (std::is_same<glm::u16vec3, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_SHORT3;
		else if (std::is_same<glm::u16vec4, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_SHORT4;
		else if (std::is_same<long long, T>::value)
			rtFormat = RT_FORMAT_LONG_LONG;
		else if (std::is_same<glm::i64vec2, T>::value)
			rtFormat = RT_FORMAT_LONG_LONG2;
		else if (std::is_same<glm::i64vec3, T>::value)
			rtFormat = RT_FORMAT_LONG_LONG3;
		else if (std::is_same<glm::i64vec4, T>::value)
			rtFormat = RT_FORMAT_LONG_LONG4;
		else if (std::is_same<unsigned long long, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_LONG_LONG;
		else if (std::is_same<glm::u64vec2, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_LONG_LONG2;
		else if (std::is_same<glm::u64vec3, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_LONG_LONG3;
		else if (std::is_same<glm::u64vec4, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_LONG_LONG4;
		else if (std::is_same<half, T>::value)
			rtFormat = RT_FORMAT_HALF;
		else if (std::is_same<glm::vec<2, half>, T>::value)
			rtFormat = RT_FORMAT_HALF2;
		else if (std::is_same<glm::vec<3, half>, T>::value)
			rtFormat = RT_FORMAT_HALF3;
		else if (std::is_same<glm::vec<4, half>, T>::value)
			rtFormat = RT_FORMAT_HALF4;
		else if (std::is_same<char, T>::value)
			rtFormat = RT_FORMAT_BYTE;
		else if (std::is_same<glm::vec<2, char>, T>::value)
			rtFormat = RT_FORMAT_BYTE2;
		else if (std::is_same<glm::vec<3, char>, T>::value)
			rtFormat = RT_FORMAT_BYTE3;
		else if (std::is_same<glm::vec<4, char>, T>::value)
			rtFormat = RT_FORMAT_BYTE4;
		else if (std::is_same<unsigned char, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_BYTE;
		else if (std::is_same<glm::vec<2, unsigned char>, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_BYTE2;
		else if (std::is_same<glm::vec<3, unsigned char>, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_BYTE3;
		else if (std::is_same<glm::vec<4, unsigned char>, T>::value)
			rtFormat = RT_FORMAT_UNSIGNED_BYTE4;

		return rtFormat;
	}

	OptiXCUDABuffer(optix::Context context, rfw::utils::ArrayProxy<size_t> dimensions)
		: m_Context(context), m_Dimensions(uint(dimensions.size())), m_Width(uint(dimensions[0]))
	{
		m_Format = parse_format();

		assert(dimensions.size() <= 2 && dimensions.size() >= 1);
		switch (m_Dimensions)
		{
		case (1):
			m_Width = uint(dimensions[0]);
			if (bufferType == Read)
			{
				m_Buffer = context->createBufferForCUDA((unsigned int)(bufferType), m_Format);
				if (m_Format == RT_FORMAT_USER)
					m_Buffer->setElementSize(RTsize(sizeof(T)));
				m_Buffer->setSize(m_Width);
				CheckCUDA(cudaMalloc(&m_DevicePointer, m_Width * sizeof(T)));
				m_CUDAAllocated = true;
				m_Buffer->setDevicePointer(0, m_DevicePointer);
			}
			else
			{
				m_Buffer = context->createBufferForCUDA((unsigned int)(bufferType), m_Format, m_Width);
				if (m_Format == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				m_DevicePointer = (T *)(m_Buffer->getDevicePointer(0));
			}

			break;
		case (2):
			m_Width = uint(dimensions[0]);
			m_Height = uint(dimensions[1]);
			if (bufferType == Read)
			{
				m_Buffer = context->createBufferForCUDA((unsigned int)(bufferType), m_Format);
				if (m_Format == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				m_Buffer->setSize(m_Width, m_Height);
				CheckCUDA(cudaMallocPitch(&m_DevicePointer, &m_Pitch, m_Width, m_Height));
				m_CUDAAllocated = true;
				m_Buffer->setDevicePointer(0, m_DevicePointer);
			}
			else
			{
				m_Buffer = context->createBufferForCUDA((unsigned int)(bufferType), m_Format, m_Width, m_Height);
				if (m_Format == RT_FORMAT_USER)
					m_Buffer->setElementSize(sizeof(T));
				m_DevicePointer = (T *)(m_Buffer->getDevicePointer(0));
			}
			break;
		default:
			assert(false);
		}
		assert(m_DevicePointer);
		m_Buffer->validate();
	}

	OptiXCUDABuffer(optix::Context context, size_t size) : OptiXCUDABuffer(context, rfw::utils::ArrayProxy<size_t>(size)) {}
	OptiXCUDABuffer(optix::Context context, size_t width, size_t height) : OptiXCUDABuffer(context, rfw::utils::ArrayProxy<size_t>({width, height})) {}

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

	void copy_to_device(const T *data, rfw::utils::ArrayProxy<size_t> dimensions, size_t offset = 0, bool async = false)
	{
		assert(m_CUDAAllocated);
		assert(dimensions.size() <= 2 && dimensions.size() >= 1);
		if (dimensions.size() > m_Dimensions)
			throw std::runtime_error("Dimensions are bigger than buffer dimensions");

		switch (dimensions.size())
		{
		case (1):
			if (async)
			{
				CheckCUDA(cudaMemcpyAsync(m_DevicePointer + offset, data, dimensions[0] * sizeof(T), cudaMemcpyHostToDevice));
			}
			else
			{
				CheckCUDA(cudaMemcpy(m_DevicePointer + offset, data, dimensions[0] * sizeof(T), cudaMemcpyHostToDevice));
			}
			break;
		case (2):
			if (async)
			{
				CheckCUDA(cudaMemcpy2DAsync(m_DevicePointer + offset, m_Pitch, data, dimensions[0] * sizeof(T), dimensions[0] * sizeof(T), dimensions[1],
											cudaMemcpyHostToDevice));
			}
			else
			{
				CheckCUDA(cudaMemcpy2D(m_DevicePointer + offset, m_Pitch, data, dimensions[0] * sizeof(T), dimensions[0] * sizeof(T), dimensions[1],
									   cudaMemcpyHostToDevice));
			}
			break;
		default:
			assert(false);
		}
		m_Buffer->validate();
	}

	void copy_to_device(rfw::utils::ArrayProxy<T> data, size_t offset = 0, bool async = false) { copy_to_device(data.data(), {data.size()}, offset, async); }

	void clear()
	{
		assert(!m_CUDAAllocated);
		switch (m_Dimensions)
		{
		case (1):
			CheckCUDA(cudaMemset(m_DevicePointer, 0, m_Width * m_Height * sizeof(T)));
			break;
		case (2):
			CheckCUDA(cudaMemset2D(m_DevicePointer, m_Pitch, m_Width * sizeof(T), m_Width * sizeof(T), m_Height));
			break;
		default:
			assert(false);
		}

		m_Buffer->validate();
	}

	void clear_async()
	{
		assert(!m_CUDAAllocated);
		switch (m_Dimensions)
		{
		case (1):
			CheckCUDA(cudaMemsetAsync(m_DevicePointer, 0, size() * sizeof(T)));
			break;
		case (2):
			CheckCUDA(cudaMemset2DAsync(m_DevicePointer, m_Pitch, m_Width * sizeof(T), m_Width * sizeof(T), m_Height));
			break;
		default:
			assert(false);
		}
		m_Buffer->validate();
	}

	size_t byte_size() const { return size() * sizeof(T); }

	size_t size() const
	{
		switch (m_Dimensions)
		{
		case (1):
			return m_Width;
		case (2):
			return m_Width * m_Height;
		default:
			assert(false);
			return 0;
		}
	}

	size_t width() const { return m_Width; }

	size_t height() const { return m_Height; }

	size_t dimensions() const { return m_Dimensions; }

	T *device_data() { return m_DevicePointer; }

	optix::Buffer buffer() const { return m_Buffer; }

	const T *device_data() const { return m_DevicePointer; }

	T *map()
	{
		m_Buffer->validate();
		return (T *)m_Buffer->map();
	}

	void unmap()
	{
		m_Buffer->unmap();
		m_Buffer->validate();
	}

  private:
	bool m_CUDAAllocated = false;
	optix::Context m_Context;
	size_t m_Pitch = 0;
	unsigned int m_Dimensions = 0;
	unsigned int m_Width = 0;
	unsigned int m_Height = 0;

	T *m_DevicePointer = nullptr;
	RTformat m_Format;
	optix::Buffer m_Buffer = nullptr;
};
