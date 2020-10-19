#pragma once
#include <GL/glew.h>

#include <glm/glm.hpp>
#include <glm/matrix.hpp>

#include <string>
#include "check.h"

#include <rfw/utils/file.h>

#include "check.h"

namespace rfw::utils
{

class shader
{
  public:
	/**
	 * Initializes a shader program
	 * @param vertexPath
	 * @param fragmentPath
	 */
	explicit shader(std::string vertexPath, std::string fragmentPath);
	~shader();

	/**
	 * Returns shader program ID
	 * @return
	 */
	const GLuint &get_shader_id() const { return m_ShaderId; }

	/**
	 * Use this shader
	 */
	void bind() const { glUseProgram(m_ShaderId); }

	/**
	 * Unbind any shader
	 */
	void unbind() const { glUseProgram(0); }

	void set_uniform(const char *name, const float *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform1fv(location, count, value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec2 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform2fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec3 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform3fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec4 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform4fv(location, count, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const int *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform1iv(location, count, value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec2 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform2iv(location, count, (GLint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec3 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform3iv(location, count, (GLint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec4 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform4iv(location, count, (GLint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const unsigned int *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform1uiv(location, count, value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec2 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform2uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec3 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform3uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec4 *value, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniform4uiv(location, count, (GLuint *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat2 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix2fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat3 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix3fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat4 *value, GLsizei count, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix4fv(location, count, transpose, (GLfloat *)value);
		CheckGL();
	}

	void set_uniform(const char *name, const float &value)
	{
		const auto location = get_uniform_location(name);
		glUniform1fv(location, 1, &value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec2 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform2fv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec3 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform3fv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::vec4 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform4fv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const int &value)
	{
		const auto location = get_uniform_location(name);
		glUniform1iv(location, 1, &value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec2 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform2iv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec3 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform3iv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::ivec4 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform4iv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const unsigned int &value)
	{
		const auto location = get_uniform_location(name);
		glUniform1uiv(location, 1, &value);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec2 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform2uiv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec3 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform3uiv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::uvec4 &value)
	{
		const auto location = get_uniform_location(name);
		glUniform4uiv(location, 1, &value.x);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat2 &value, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix2fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat3 &value, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix3fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void set_uniform(const char *name, const glm::mat4 &value, const bool transpose = false)
	{
		const auto location = get_uniform_location(name);
		glUniformMatrix4fv(location, 1, transpose, &value[0][0]);
		CheckGL();
	}

	void setHandle(const char *name, const uint64_t handle)
	{
		const auto location = get_uniform_location(name);
		glUniformHandleui64ARB(location, handle);
		CheckGL();
	}

	void setHandles(const char *name, const uint64_t *handles, GLsizei count)
	{
		const auto location = get_uniform_location(name);
		glUniformHandleui64vARB(location, count, handles);
		CheckGL();
	}

	/**
	 * Return uniform location at 'name'
	 * @param name		Uniform name
	 * @param index		Index of uniform array, -1 if uniform is not an array
	 * @return			Uniform location
	 */
	GLint get_uniform_location(const char *name, int index = -1) const
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
	GLint get_attribute_location(const char *name) const
	{
		return glGetAttribLocation(m_ShaderId, name);
		CheckGL();
	}

  private:
	GLuint m_ShaderId;
	std::string m_VertPath;
	std::string m_FragPath;

	GLuint load();
	void check_compile_errors(const char *file, GLuint shader, const std::string &type) const;
};
} // namespace rfw::utils