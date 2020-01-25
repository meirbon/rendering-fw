#pragma once
#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>

#include <iostream>
#include <string>
#include <vector>
#include "CheckGL.h"

#include <utils/File.h>

namespace rfw::utils
{

class GLShader
{
  public:
	/**
	 * Initializes a shader program
	 * @param vertexPath
	 * @param fragmentPath
	 */
	explicit GLShader(std::string vertexPath, std::string fragmentPath);
	~GLShader();

	/**
	 * Returns shader program ID
	 * @return
	 */
	const GLuint &getShaderId() const { return m_ShaderId; }

	/**
	 * Use this shader
	 */
	void bind() const { glUseProgram(m_ShaderId); }

	/**
	 * Unbind any shader
	 */
	void unbind() const { glUseProgram(0); }

	void setUniform(const char *name, const float *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform1fv(location, count, value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec2 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform2fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec3 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform3fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec4 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform4fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const int *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform1iv(location, count, value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec2 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform2iv(location, count, (GLint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec3 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform3iv(location, count, (GLint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec4 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform4iv(location, count, (GLint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const unsigned int *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform1uiv(location, count, value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec2 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform2uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec3 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform3uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec4 *value, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniform4uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat2 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix2fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat3 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix3fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat4 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix4fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void setUniform(const char *name, const float &value)
	{
		const auto location = getUniformLocation(name);
		glUniform1fv(location, 1, &value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec2 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform2fv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec3 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform3fv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::vec4 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform4fv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const int &value)
	{
		const auto location = getUniformLocation(name);
		glUniform1iv(location, 1, &value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec2 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform2iv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec3 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform3iv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::ivec4 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform4iv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const unsigned int &value)
	{
		const auto location = getUniformLocation(name);
		glUniform1uiv(location, 1, &value);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec2 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform2uiv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec3 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform3uiv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::uvec4 &value)
	{
		const auto location = getUniformLocation(name);
		glUniform4uiv(location, 1, &value.x);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat2 &value, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix2fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat3 &value, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix3fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void setUniform(const char *name, const glm::mat4 &value, const bool transpose = false)
	{
		const auto location = getUniformLocation(name);
		glUniformMatrix4fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void setHandle(const char *name, const uint64_t handle)
	{
		const auto location = getUniformLocation(name);
		glUniformHandleui64ARB(location, handle);
		CheckGL();
	}

	void setHandles(const char *name, const uint64_t *handles, GLsizei count)
	{
		const auto location = getUniformLocation(name);
		glUniformHandleui64vARB(location, count, handles);
		CheckGL();
	}

	/**
	 * Return uniform location at 'name'
	 * @param name		Uniform name
	 * @param index		Index of uniform array, -1 if uniform is not an array
	 * @return			Uniform location
	 */
	GLint getUniformLocation(const char *name, int index = -1)
	{
		if (index >= 0)
		{
			char n[64]{};
			std::sprintf(n, "%s[%i]", name, index);
			return glGetUniformLocation(m_ShaderId, n);
		}

		return glGetUniformLocation(m_ShaderId, name);
	}

	/**
	 * Return attribute location at 'name'
	 * @param name		Attribute name
	 * @return			Attribute location
	 */
	GLint getAttributeLocation(const char *name)
	{
		return glGetAttribLocation(m_ShaderId, name);
		CheckGL();
	}

  private:
	GLuint m_ShaderId;
	std::string m_VertPath;
	std::string m_FragPath;

	GLuint load();
	void checkCompileErrors(const char *file, GLuint shader, const std::string &type) const;
};
} // namespace rfw::utils