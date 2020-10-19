#pragma once

#include <functional>

#include <rfw/utils/lib_export.h>
#include <rfw/context/export.h>

typedef rfw::RenderContext *(*CreateContextFunction)();
typedef void (*DestroyContextFunction)(rfw::RenderContext *);

#define CREATE_RENDER_CONTEXT_FUNC_NAME "createRenderContext"
#define DESTROY_RENDER_CONTEXT_FUNC_NAME "destroyRenderContext"

extern "C" RENDER_API rfw::RenderContext *createRenderContext();
extern "C" RENDER_API void destroyRenderContext(rfw::RenderContext *ptr);
