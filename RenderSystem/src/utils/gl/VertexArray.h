#pragma once

#include <GL/glew.h>
#include "CheckGL.h"

#include "GLBuffer.h"

namespace rfw::utils
{
class GLVertexArray
{
  public:
	struct Binding
	{
		GLuint bufferID;
		GLuint bufferSize;
	};

	GLVertexArray() { glGenVertexArrays(1, &m_ID); }
	~GLVertexArray() { glDeleteVertexArrays(1, &m_ID); }
	GLVertexArray(const GLVertexArray &other) = delete;

	void bind() const { glBindVertexArray(m_ID); }
	void unbind() const { glBindVertexArray(0); }

	template <typename T, GLenum B>
	void setBuffer(GLuint index, const rfw::utils::Buffer<T, GL_ARRAY_BUFFER, B> &buffer, GLint elementCount = -1, GLenum type = GL_FLOAT,
				   bool normalized = false, size_t structOffset = 0) const
	{
		if (elementCount < 0)
			elementCount = sizeof(T) / 4;
		assert(elementCount >= 1);

		bind();
		buffer.bind();
		glEnableVertexAttribArray(index);
		glVertexAttribPointer(index, elementCount, type, normalized, sizeof(T), (const void *)structOffset);
		unbind();
		CheckGL();
	}

	std::vector<std::pair<GLuint, Binding>> getBindingInfo()
	{
		std::vector<std::pair<GLuint, Binding>> result;

		GLint vertexAttribCount = 0;
		glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &vertexAttribCount);
		for (unsigned i = 0; i < vertexAttribCount; ++i)
		{
			int isOn = 0;
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &isOn);

			if (isOn)
			{
				int VBOID = 0;
				int size = 0;

				glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &VBOID);
				glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);

				Binding b;
				b.bufferID = VBOID;
				b.bufferSize = size;
				result.push_back(std::make_pair(GLuint(i), b));
			}
		}

		return result;
	}

  private:
	GLuint m_ID;
};
} // namespace rfw::utils