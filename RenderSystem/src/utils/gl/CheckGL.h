#pragma once

#include <GL/glew.h>
#include <utils/Logger.h>

namespace rfw::utils
{
static void _CheckGL(const char *f, int l)
{
	GLenum error = glGetError();
	while (error != GL_NO_ERROR)
	{
		logger::warning(f, l, "%i :: %s", error, glewGetErrorString(error));
		error = glGetError();
	}
}

#define CheckGL() rfw::utils::_CheckGL(__FILE__, __LINE__)

} // namespace rfw::utils