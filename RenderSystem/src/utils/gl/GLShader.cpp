#include <GL/glew.h>

#include "GLShader.h"

#include <string>
#include <vector>

#include "utils/File.h"
#include "utils/Logger.h"

using namespace rfw;
using namespace utils;

GLShader::GLShader(const char *vertexPath, const char *fragmentPath) : m_VertPath(vertexPath), m_FragPath(fragmentPath)
{
	m_ShaderId = load();
}

GLuint GLShader::load()
{
	const GLuint program = glCreateProgram();
	const GLuint vert_shader = glCreateShader(GL_VERTEX_SHADER);
	const GLuint frag_shader = glCreateShader(GL_FRAGMENT_SHADER);

	std::string vert_string = utils::file::read(m_VertPath);
	std::string frag_string = utils::file::read(m_FragPath);

	const char *vert_source = vert_string.c_str();
	const char *frag_source = frag_string.c_str();

	glShaderSource(vert_shader, 1, &vert_source, nullptr);
	glCompileShader(vert_shader);
	checkCompileErrors(vert_shader, "VERTEX");

	glShaderSource(frag_shader, 1, &frag_source, nullptr);
	glCompileShader(frag_shader);
	checkCompileErrors(frag_shader, "FRAGMENT");

	glAttachShader(program, vert_shader);
	glAttachShader(program, frag_shader);

	glLinkProgram(program);
	glValidateProgram(program);
	checkCompileErrors(program, "PROGRAM");

	glDeleteShader(vert_shader);
	glDeleteShader(frag_shader);

	return program;
}

void GLShader::checkCompileErrors(GLuint shader, std::string type)
{
	GLint success;
	std::vector<GLchar> infoLog(2048);
	if (type != "PROGRAM")
	{
		glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
		if (!success)
		{
			GLint maxLength = 0;
			glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
			if (maxLength > 0)
				infoLog.resize(maxLength);
			glGetShaderInfoLog(shader, (GLsizei)infoLog.size(), nullptr, infoLog.data());
			std::cout << "Error: " << infoLog.data() << std::endl;
			FAILURE("GLShader compilation error of type: %s\n%s", type.c_str(), infoLog.data());
		}
	}
	else
	{
		glGetProgramiv(shader, GL_LINK_STATUS, &success);
		if (!success)
		{
			GLint maxLength = 0;
			glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &maxLength);
			if (maxLength > 0)
				infoLog.resize(maxLength);
			glGetProgramInfoLog(shader, (GLsizei)infoLog.size(), nullptr, infoLog.data());
			std::cout << "Error: " << infoLog.data() << std::endl;
			FAILURE("GLShader compilation error of type: %s\nLog: %s", type.c_str(), infoLog.data());
		}
	}
}

GLShader::~GLShader() { glDeleteProgram(m_ShaderId); }