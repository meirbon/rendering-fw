#pragma once

#include <GL/glew.h>

#include <vector>
#include <tuple>

#include "check.h"
#include "buffer.h"

namespace rfw::utils
{
class vertex_array
{
  public:
	struct binding
	{
		GLuint bufferID;
		GLuint bufferSize;
	};

	vertex_array()
	{
		glGenVertexArrays(1, &m_ID);
		CheckGL();
	}
	~vertex_array() { glDeleteVertexArrays(1, &m_ID); }
	vertex_array(const vertex_array &other) = delete;

	void bind() const
	{
		glBindVertexArray(m_ID);
		CheckGL();
	}

	static void unbind()
	{
		glBindVertexArray(0);
		CheckGL();
	}

	template <typename T, GLenum B>
	void set_buffer(GLuint index, const rfw::utils::buffer<T, GL_ARRAY_BUFFER, B> &buffer, GLint elementCount = -1,
					GLenum type = GL_FLOAT, bool normalized = false, size_t structOffset = 0) const
	{
		if (elementCount < 0)
			elementCount = sizeof(T) / 4;
		assert(elementCount >= 1);

		bind();
		buffer.bind();
		glEnableVertexAttribArray(index);
		CheckGL();
		glVertexAttribPointer(index, elementCount, type, normalized, sizeof(T), (const void *)structOffset);
		CheckGL();
		unbind();
		CheckGL();
	}

	std::vector<std::pair<GLuint, binding>> getBindingInfo()
	{
		std::vector<std::pair<GLuint, binding>> result;

		bind();

		GLint vertexAttribCount = 0;
		glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &vertexAttribCount);
		for (int i = 0; i < vertexAttribCount; ++i)
		{
			int isOn = 0;
			glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_ENABLED, &isOn);

			if (isOn)
			{
				auto VBOID = 0;
				auto size = 0;

				glGetVertexAttribiv(i, GL_VERTEX_ATTRIB_ARRAY_BUFFER_BINDING, &VBOID);
				glGetBufferParameteriv(GL_ARRAY_BUFFER, GL_BUFFER_SIZE, &size);

				binding b = {static_cast<GLuint>(VBOID), static_cast<GLuint>(size)};
				result.emplace_back(GLuint(i), b);
			}
		}

		unbind();

		return result;
	}

  private:
	GLuint m_ID{};
};
} // namespace rfw::utils