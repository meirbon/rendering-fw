//
// Created by Mèir Noordermeer on 2019-08-20.
//

#ifndef RENDERING_FW_SRC_UTILS_STRING_HPP
#define RENDERING_FW_SRC_UTILS_STRING_HPP

#include <vector>
#include <cstdarg>
#include <cstdio>
#include <cstring>

namespace rfw
{
namespace utils
{
namespace string
{

static void format(std::vector<char> &buffer, const char *format, va_list args)
{
	if (buffer.empty()) buffer.resize(8192);
#ifdef _WIN32
	vsprintf_s(buffer.data(), buffer.size(), format, args);
#else
	vsprintf(buffer.data(), format, args);
#endif
}

static bool ends_with(const std::string_view &value, const std::string_view &ending)
{
	if (ending.size() > value.size())
		return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

} // namespace string
} // namespace utils
} // namespace rfw
#endif // RENDERING_FW_SRC_UTILS_STRING_HPP
