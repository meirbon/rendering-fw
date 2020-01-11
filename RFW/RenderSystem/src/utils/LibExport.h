#pragma once

#ifdef RENDER_API_EXPORT
#ifdef WIN32
#define RENDER_API __declspec(dllexport)
#else
#define RENDER_API __attribute__((visibility("default")))
#endif
#else
#ifdef WIN32
#define RENDER_API __declspec(dllimport)
#else
#define RENDER_API
#endif
#endif