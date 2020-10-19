#pragma once

#include <GL/glew.h>

#include <cassert>

#include "check.h"

namespace rfw::utils
{

template <typename T, GLenum B, GLenum U> class buffer
{
  public:
	buffer()
	{
		glGenBuffers(1, &m_BufferId);
		CheckGL();
	}

	buffer(const std::vector<T> &data)
	{
		glGenBuffers(1, &m_BufferId);
		CheckGL();
		set_data(data);
	}
	buffer(const T *data, size_t elementCount)
	{
		glGenBuffers(1, &m_BufferId);
		CheckGL();
		set_data(data, elementCount * sizeof(T));
	}

	buffer(const buffer<T, B, U> &other) = delete;

	~buffer() { glDeleteBuffers(1, &m_BufferId); }

	GLuint getID() const { return m_BufferId; }
	void bind() const
	{
		glBindBuffer(B, m_BufferId);
		CheckGL();
	}
	void unbind() const { glBindBuffer(B, 0); }
	size_t getSize() const { return m_Size; }
	size_t get_count() const { return m_Size / sizeof(T); }
	void set_data(const std::vector<T> &data) { set_data(data.data(), data.size() * sizeof(T)); }
	void set_data(const void *data, size_t sizeInBytes)
	{
		glBindBuffer(B, m_BufferId);
		glBufferData(B, sizeInBytes, data, U);
		glBindBuffer(B, 0);
		m_Size = sizeInBytes;
		CheckGL();
	}

	T *map(GLenum mode = GL_WRITE_ONLY) const
	{
		bind();
		assert(mode == GL_WRITE_ONLY || mode == GL_READ_ONLY || mode == GL_READ_WRITE);
		GLvoid *p = glMapBuffer(B, GL_WRITE_ONLY);
		return reinterpret_cast<T *>(p);
	}

	void unmap() const
	{
		bind();
		glUnmapBuffer(B);
		unbind();
	}

	std::vector<T> read() const
	{
		if (m_Size == 0)
			return {};

		std::vector<T> data(m_Size / sizeof(T));
		memcpy(data.data(), map(), m_Size);
		unmap();
		return data;
	}

  private:
	GLuint m_BufferId = 0;
	size_t m_Size = 0;
};
} // namespace rfw::utils