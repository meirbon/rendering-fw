#pragma once

// Include files in this header
// Only include this header in every file for faster compilation

#include <rfw.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>
#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>

#include <GL/glew.h>
#if defined(WIN32)
#include <Windows.h>
#endif

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION 1
#include <GL/glew.h>
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl_ext.h>
#include <OpenGL/CGLCurrent.h>
#include <OpenGL/CGLDevice.h>
#else
#include <GL/glew.h>
#ifdef __linux__
#include <GL/glx.h>
#endif
#include <CL/cl.h>
#include <CL/cl_gl_ext.h>
#endif

#include <MathIncludes.h>

#include "CL/CheckCL.h"

#include "CL/OpenCL.h"

#include "Context.h"