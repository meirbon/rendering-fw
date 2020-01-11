#pragma once

#include <algorithm>
#include <iostream>
#include <tuple>
#include <array>
#include <memory>

#include <utils/ArrayProxy.h>

#ifdef __APPLE__
#ifndef CL_SILENCE_DEPRECATION
#define CL_SILENCE_DEPRECATION
#endif
#include <GL/glew.h>
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLDevice.h>
#else
#include <GL/glew.h>
#ifdef __linux__
#include <GL/glx.h>
#endif
#include <CL/cl.h>
#include <CL/cl_gl_ext.h>
#endif

#include <glm/glm.hpp>

#include <glm/ext.hpp>

#include "CheckCL.h"

// #define USE_CPU_DEVICE

namespace cl
{

class CLContext;
enum BufferType
{
	DATA = 1,
	TEXTURE = 2,
	TARGET = 3
};

template <typename T, BufferType TYPE = DATA> class CLBuffer
{
  public:
	CLBuffer() {}
	CLBuffer(std::shared_ptr<CLContext> context, unsigned int element_count, const T *ptr);
	CLBuffer(std::shared_ptr<CLContext> context, const std::vector<T> &data);
	CLBuffer(std::shared_ptr<CLContext> context, GLuint texID, GLuint width, GLuint height);
	~CLBuffer() { clReleaseMemObject(m_DeviceBuffer); }

	size_t size_in_bytes() const { return m_HostBuffer.size() * sizeof(T); }
	size_t size() const { return m_HostBuffer.size(); }

	T &at(size_t idx) { return m_HostBuffer[idx]; }
	const T &at(size_t idx) const { return m_HostBuffer[idx]; }

	T &operator[](size_t idx) { return m_HostBuffer[idx]; }
	const T &operator[](size_t idx) const { return m_HostBuffer[idx]; }

	const cl_mem *get_device_buffer() const { return &m_DeviceBuffer; }

	void copy_to_device(bool blocking = true);

	void copy_to_host(bool blocking = true);

	void copy_to(cl_mem buffer);

	template <typename U, BufferType K> void copy_to(CLBuffer<U, K> &other) const { copy_to(other.get_device_buffer()); }

	void clear(unsigned int value = 0);

	void read(void *dst);

	template <typename U> void write(rfw::utils::ArrayProxy<U> data)
	{
		assert(data.size() <= m_HostBuffer.size() * sizeof(T));
		write(data.data());
	}

	void write(void *dst);

  private:
	std::shared_ptr<CLContext> m_Context;
	cl_mem m_DeviceBuffer = nullptr;
	cl_mem m_PinnedBuffer = nullptr;
	GLuint m_TexID = 0;
	std::vector<T> m_HostBuffer;
};

class CLKernel
{
  public:
	// constructor / destructor
	CLKernel(std::shared_ptr<CLContext> context, const char *file, const char *entryPoint, std::array<size_t, 3> workSize, std::array<size_t, 3> localSize);
	~CLKernel();

	cl_kernel get_kernel() const { return m_Kernel; }
	cl_program get_program() const { return m_Program; }

	void run();

	template <typename T> void run(CLBuffer<T, TARGET> &buffer) const;
	template <typename T> void run(CLBuffer<T, TEXTURE> &buffer) const;

	template <typename T> void set_argument(const cl_uint idx, const T val)
	{
		if (std::is_pointer<T>::value)
			assert(false);

		static_assert(sizeof(T) % 4 == 0, "CLKernel argument objects should be 4 byte aligned");
		clSetKernelArg(m_Kernel, idx, sizeof(T), &val);
	}

	template <typename T, BufferType U> void set_buffer(const cl_uint idx, const CLBuffer<T, U> &buffer) const
	{
		clSetKernelArg(m_Kernel, idx, sizeof(cl_mem *), buffer.get_device_buffer());
	}

	template <typename T, BufferType U> void set_buffer(const cl_uint idx, const CLBuffer<T, U> *buffer) const
	{
		clSetKernelArg(m_Kernel, idx, sizeof(cl_mem *), buffer->get_device_buffer());
	}

	size_t get_dimensions() const { return m_Dimensions; }

	const size_t *get_offset() const { return m_Offset.data(); }

	const size_t *get_work_size() const { return m_WorkSize.data(); }

	const size_t *get_local_size() const { return m_LocalSize.data(); }

	void set_offset(const std::array<size_t, 3> &offset);

	void set_global_size(const std::array<size_t, 3>& global_size);

	void set_work_size(const std::array<size_t, 3> &work_size);

	void set_local_size(const std::array<size_t, 3> &local_size);

  private:
	std::shared_ptr<CLContext> m_Context;
	cl_kernel m_Kernel;
	cl_program m_Program;

	size_t m_Dimensions = 0;
	std::array<size_t, 3> m_Offset = {0, 0, 0};
	std::array<size_t, 3> m_WorkSize = {0, 0, 0};
	std::array<size_t, 3> m_LocalSize = {0, 0, 0};
};

class CLContext
{
  public:
	CLContext();
	~CLContext();

	void submit(CLKernel &kernel) const;

	template <typename T> void submit(CLBuffer<T, TARGET> &buffer, CLKernel &kernel) const
	{
		assert(m_CanDoInterop);
		glFinish();
		CheckCL(clEnqueueAcquireGLObjects(m_Queue, 1, buffer.get_device_buffer(), 0, 0, 0));
		CheckCL(clEnqueueNDRangeKernel(m_Queue, kernel.get_kernel(), kernel.get_dimensions(), kernel.get_offset(), kernel.get_work_size(),
									   kernel.get_local_size(), 0, 0, 0));
		CheckCL(clEnqueueReleaseGLObjects(m_Queue, 1, buffer->GetDevicePtr(), 0, 0, 0));
	}

	template <typename T> void submit(CLBuffer<T, TEXTURE> &buffer, CLKernel &kernel) const
	{
		assert(m_CanDoInterop);
		glFinish();
		CheckCL(clEnqueueAcquireGLObjects(m_Queue, 1, buffer.get_device_buffer(), 0, 0, 0));
		CheckCL(clEnqueueNDRangeKernel(m_Queue, kernel.get_kernel(), kernel.get_dimensions(), kernel.get_offset(), kernel.get_work_size(),
									   kernel.get_local_size(), 0, 0, 0));
		CheckCL(clEnqueueReleaseGLObjects(m_Queue, 1, buffer->GetDevicePtr(), 0, 0, 0));
	}

	void finish() const;

	cl_context get_context() const { return m_Context; }
	cl_command_queue get_queue() const { return m_Queue; }
	cl_device_id get_device() const { return m_Device; }

	bool is_interop() const { return m_CanDoInterop; }

  private:
	bool m_CanDoInterop = false;
	bool init_cl();

	cl_context m_Context = nullptr;
	cl_command_queue m_Queue = nullptr;
	cl_device_id m_Device = nullptr;
};

template <typename T, BufferType TYPE>
CLBuffer<T, TYPE>::CLBuffer(std::shared_ptr<CLContext> context, unsigned int element_count, const T *ptr) : m_Context(context)
{
	m_DeviceBuffer = clCreateBuffer(context->get_context(), CL_MEM_READ_WRITE, element_count * sizeof(T), 0, 0);

	if (ptr)
		m_HostBuffer = std::vector<T>(ptr, ptr + element_count);
	else
		m_HostBuffer.resize(element_count);
}
template <typename T, BufferType TYPE>
CLBuffer<T, TYPE>::CLBuffer(std::shared_ptr<CLContext> context, const std::vector<T> &data) : m_Context(context), m_HostBuffer(data)
{
	m_DeviceBuffer = clCreateBuffer(context->get_context(), CL_MEM_READ_WRITE, m_HostBuffer.size() * sizeof(T), 0, 0);
}
template <typename T, BufferType TYPE>
CLBuffer<T, TYPE>::CLBuffer(std::shared_ptr<CLContext> context, GLuint texID, GLuint width, GLuint height) : m_Context(context)
{
	assert(texID != 0);
	m_TexID = texID;

	if (context->is_interop())
	{
		if constexpr (TYPE == TARGET)
			m_DeviceBuffer = clCreateFromGLTexture(context->get_context(), CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, texID, 0);
		else
			m_DeviceBuffer = clCreateFromGLTexture(context->get_context(), CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, texID, 0);
	}
	else
	{
		// can't directly generate buffer from texture
		m_HostBuffer.resize(width * height * sizeof(float) * 4);
		cl_image_format format;
		cl_image_desc desc;
		desc.image_type = CL_MEM_OBJECT_IMAGE2D;
		desc.image_width = width;
		desc.image_height = height;
		desc.image_depth = 0;
		desc.image_array_size = 0;
		desc.image_row_pitch = 0;
		desc.image_slice_pitch = 0;
		desc.num_mip_levels = 0;
		desc.num_samples = 0;
		desc.buffer = 0;
		format.image_channel_order = CL_RGBA;
		format.image_channel_data_type = CL_FLOAT;

		if constexpr (TYPE == TARGET)
			m_DeviceBuffer = clCreateImage(context->get_context(), CL_MEM_WRITE_ONLY, &format, &desc, m_HostBuffer.data(), 0);
		else
			m_DeviceBuffer = clCreateImage(context->get_context(), CL_MEM_READ_WRITE, &format, &desc, m_HostBuffer.data(), 0);
	}
}
template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::copy_to_device(bool blocking)
{
	CheckCL(clEnqueueWriteBuffer(m_Context->get_queue(), m_DeviceBuffer, blocking, 0, m_HostBuffer.size() * sizeof(T), m_HostBuffer.data(), 0, 0, 0));
}

template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::copy_to_host(bool blocking)
{
	CheckCL(clEnqueueReadBuffer(m_Context->get_queue(), m_DeviceBuffer, blocking, 0, m_HostBuffer.size() * sizeof(T), m_HostBuffer.data(), 0, 0, 0));
}

template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::copy_to(cl_mem buffer)
{
	CheckCL(clEnqueueCopyBuffer(m_Context->get_queue(), m_DeviceBuffer, buffer, 0, 0, m_HostBuffer.size() * sizeof(T), 0, 0, 0));
}

template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::clear(unsigned int value)
{
	CheckCL(clEnqueueFillBuffer(m_Context->get_queue(), m_DeviceBuffer, &value, 4, 0, m_HostBuffer.size() * sizeof(T), 0, nullptr, nullptr));
}

template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::read(void *dst)
{
	auto *data = static_cast<unsigned char *>(
		clEnqueueMapBuffer(m_Context->get_queue(), m_PinnedBuffer, CL_TRUE, CL_MAP_READ, 0, m_HostBuffer.size() * sizeof(T), 0, nullptr, nullptr, nullptr));
	clEnqueueReadBuffer(m_Context->get_queue(), m_DeviceBuffer, CL_TRUE, 0, m_HostBuffer.size() * sizeof(T), data, 0, nullptr, nullptr);
	memcpy(dst, data, m_HostBuffer.size() * sizeof(T));
	clEnqueueUnmapMemObject(m_Context->get_queue(), m_PinnedBuffer, nullptr, 0, 0, 0);
	m_PinnedBuffer = nullptr;
}
template <typename T, BufferType TYPE> void CLBuffer<T, TYPE>::write(void *dst)
{
	auto *data = static_cast<unsigned char *>(
		clEnqueueMapBuffer(m_Context->get_queue(), m_PinnedBuffer, CL_TRUE, CL_MAP_WRITE, 0, m_HostBuffer.size() * sizeof(T), 0, nullptr, nullptr, nullptr));
	memcpy(data, dst, m_HostBuffer.size() * sizeof(T));
	clEnqueueWriteBuffer(m_Context->get_queue(), m_DeviceBuffer, CL_FALSE, 0, m_HostBuffer.size() * sizeof(T), data, 0, nullptr, nullptr);
	clEnqueueUnmapMemObject(m_Context->get_queue(), m_PinnedBuffer, nullptr, 0, 0, 0);
	m_PinnedBuffer = nullptr;
}

template <typename T> void CLKernel::run(CLBuffer<T, TARGET> &buffer) const { m_Context->submit(buffer, *this); }

template <typename T> void CLKernel::run(CLBuffer<T, TEXTURE> &buffer) const { m_Context->submit(buffer, *this); }
} // namespace cl