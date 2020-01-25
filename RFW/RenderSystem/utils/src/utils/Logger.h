//
// Created by Mèir Noordermeer on 2019-08-20.
//

#ifndef RENDERING_FW_SRC_UTILS_LOGGER_HPP
#define RENDERING_FW_SRC_UTILS_LOGGER_HPP

#include <cstdarg>
#include <cstring>
#include <iostream>
#include "String.h"
#include <vector>

namespace rfw::utils::logger
{
constexpr size_t BUFFER_SIZE = 16384;
static void debug(const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	memset(buffer.data(), 0, sizeof(char) * buffer.size());
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cout << "DEBUG: " << buffer.data() << std::endl;
}

static void warning(const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	memset(buffer.data(), 0, sizeof(char) * buffer.size());
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cerr << "WARNING: " << buffer.data() << std::endl;
}

static void log(const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	memset(buffer.data(), 0, sizeof(char) * buffer.size());
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cout << "LOG: " << buffer.data() << std::endl;
}

static void err(const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	memset(buffer.data(), 0, sizeof(char) * buffer.size());
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cerr << "ERROR: " << buffer.data() << std::endl;
	throw std::runtime_error(buffer.data());
}

static void _debug(const char *file, int line, const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	string::format(buffer.data(), "%s::%i", file, line);
	const auto start = std::string(buffer.data());
	memset(buffer.data(), 0, buffer.size() * sizeof(char));
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cout << "DEBUG: " << start.data() << " : " << buffer.data() << std::endl;
}

static void _warning(const char *file, int line, const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	string::format(buffer.data(), "%s::%i", file, line);
	const auto start = std::string(buffer.data());
	memset(buffer.data(), 0, buffer.size() * sizeof(char));
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cerr << "WARNING: " << start.data() << " : " << buffer.data() << std::endl;
}

static void _err(const char *file, int line, const char *format, ...)
{
	std::vector<char> buffer(BUFFER_SIZE, 0);
	string::format(buffer.data(), "%s::%i", file, line);
	const auto start = std::string(buffer.data());
	memset(buffer.data(), 0, buffer.size() * sizeof(char));
	va_list arg;
	va_start(arg, format);
	string::format_list(buffer.data(), format, arg);
	va_end(arg);
	std::cerr << "ERROR: " << start.data() << " : " << buffer.data() << std::endl;
	throw std::runtime_error(buffer.data());
}

} // namespace rfw::utils::logger
/**
 * Defines a DEBUG macro to log only when the application is build in debug
 * mode.
 */
#ifndef NDEBUG
#define DEBUG(format, ...) rfw::utils::logger::_debug(__FILE__, __LINE__, format, ##__VA_ARGS__)
#else
#define DEBUG(format, ...)
#endif

#define WARNING(format, ...) rfw::utils::logger::_warning(__FILE__, __LINE__, format, ##__VA_ARGS__)
#define FAILURE(format, ...) rfw::utils::logger::_err(__FILE__, __LINE__, format, ##__VA_ARGS__)

#endif // RENDERING_FW_SRC_UTILS_LOGGER_HPP
