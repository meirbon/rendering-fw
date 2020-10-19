#pragma once

#include <cstdarg>
#include <cstdio>

#include "array_proxy.h"

namespace rfw::utils::string
{

static void format_list(char *buffer, const char *format, va_list args) { vsprintf(buffer, format, args); }

static void format(char *buffer, const char *format, ...)
{
	va_list args;
	va_start(args, format);
	vsprintf(buffer, format, args);
	va_end(args);
}

static bool begins_with(std::string_view value, std::string_view beginning)
{
	if (beginning.size() > value.size())
		return false;
	return std::equal(beginning.begin(), beginning.end(), value.begin());
}

static bool ends_with(std::string_view value, std::string_view ending)
{
	if (ending.size() > value.size())
		return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

static bool ends_with(std::string_view value, array_proxy<std::string_view> endings)
{
	for (const auto ending : endings)
	{
		if (std::equal(ending.rbegin(), ending.rend(), value.rbegin()))
			return true;
	}

	return false;
}

} // namespace rfw::utils::string
