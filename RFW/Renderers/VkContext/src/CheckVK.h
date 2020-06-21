#pragma once

#ifdef _WIN32
#include <Windows.h>
#endif
#include <vulkan/vulkan.hpp>

void _CheckResult(int line, const char *file, vk::Result x);

template <typename T> T _CheckVK(int line, const char *file, vk::ResultValue<T> v)
{
	_CheckResult(line, file, v.result);
	return v.value;
}

inline void _CheckVK(int line, const char *file, vk::Result v) { _CheckResult(line, file, v); }
inline void _CheckVK(int line, const char *file, VkResult v) { _CheckResult(line, file, static_cast<vk::Result>(v)); }

#define CheckVK(x) _CheckVK(__LINE__, __FILE__, x)
