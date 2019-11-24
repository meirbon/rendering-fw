#pragma once

#include <GL/glew.h>

namespace rfw::utils
{
template <typename T, GLenum B, GLenum U> class Buffer
{
  public:
	Buffer() { glGenBuffers(1, &m_BufferId); }

	Buffer(const std::vector<T> &data)
	{
		glGenBuffers(1, &m_BufferId);
		setData(data);
	}
	Buffer(const T *data, size_t elementCount)
	{
		glGenBuffers(1, &m_BufferId);
		setData(data, elementCount * sizeof(T));
	}

	Buffer(const Buffer<T, B, U> &other) = delete;

	~Buffer() { glDeleteBuffers(1, &m_BufferId); }

	GLuint getID() const { return m_BufferId; }
	void bind() const { glBindBuffer(B, m_BufferId); }
	void unbind() const { glBindBuffer(B, 0); }
	size_t getSize() const { return m_Size; }
	size_t getCount() const { return m_Size / sizeof(T); }
	void setData(const std::vector<T> &data)
	{
		glBindBuffer(B, m_BufferId);
		glBufferData(B, data.size() * sizeof(T), data.data(), U);
		glBindBuffer(B, 0);
		m_Size = data.size() * sizeof(T);
	}
	void setData(const void *data, size_t sizeInBytes)
	{
		glBindBuffer(B, m_BufferId);
		glBufferData(B, sizeInBytes, data, U);
		glBindBuffer(B, 0);
		m_Size = sizeInBytes;
	}

  private:
	GLuint m_BufferId = 0;
	size_t m_Size;
};
} // namespace rfw::utils