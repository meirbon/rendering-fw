#pragma once

#include <GL/glew.h>
#include <rfw/utils/logger.h>

namespace rfw::utils
{
static void _check_gl(const char *f, int l)
{
	GLenum error = glGetError();
	while (error != GL_NO_ERROR)
	{
		logger::warning(f, l, "%i :: %s", error, glewGetErrorString(error));
		error = glGetError();
	}
}

#define CheckGL() rfw::utils::_check_gl(__FILE__, __LINE__)

} // namespace rfw::utils