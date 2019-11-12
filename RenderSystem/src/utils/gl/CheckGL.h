#pragma once

#include <GL/glew.h>
#include <utils/Logger.h>

namespace rfw::utils
{
static inline void _CheckGL(const char *f, int l)
{
	GLenum error = glGetError();
	if (error != GL_NO_ERROR)
	{
		char t[1024];
		sprintf(t, "Error %i: ", error);
		if (error == 0x500)
			strcat(t, "INVALID ENUM");
		else if (error == 0x502)
			strcat(t, "INVALID OPERATION");
		else if (error == 0x501)
			strcat(t, "INVALID VALUE");
		else if (error == 0x506)
			strcat(t, "INVALID FRAMEBUFFER OPERATION");
		else
			strcat(t, "UNKNOWN ERROR");
		std::cout << "Error on line " << l << " of " << f << " :: " << t << std::endl;
	}
}

#define CheckGL() rfw::utils::_CheckGL(__FILE__, __LINE__)

} // namespace rfw::utils