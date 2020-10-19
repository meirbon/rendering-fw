#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif
#include <vulkan/vulkan.hpp>

#define CheckVK(x) _CheckVK(__LINE__, __FILE__, static_cast<vk::Result>(x))
void _CheckVK(int line, const char *file, vk::Result x);
