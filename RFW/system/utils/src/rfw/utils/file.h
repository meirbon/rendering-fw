#pragma once

#include <cassert>
#include <fstream>
#include <istream>
#include <string>

#ifdef WIN32
#include <direct.h>
#else
#include <unistd.h>
#ifdef __linux__
#include <limits.h>
#endif
#endif

#include "Logger.h"

namespace rfw::utils::file
{

static std::string get_working_path()
{
#ifdef WIN32
	const char *buffer = _getcwd(nullptr, 0);
	return std::string(buffer);
#else
	char temp[PATH_MAX];
	return (getcwd(temp, sizeof(temp)) ? std::string(temp) : std::string(""));
#endif
}

static long filesize(const std::string_view &path)
{
	std::ifstream in(path.data(), std::ifstream::binary | std::ifstream::ate);

	const auto value = in.tellg();
	return static_cast<long>(value);
}

static std::vector<char> read_binary(const std::string_view &path)
{
	std::ifstream file(path.data(), std::ios::binary);
	assert(file.is_open());
	if (!file.is_open())
	{
		WARNING("Could not open file: %s", path.data());
		return {};
	}

	const auto s = utils::file::filesize(path);
	std::vector<char> buffer(s);
	file.read(buffer.data(), s);
	file.close();
	return buffer;
}

static std::string read(const std::string_view &path)
{
	std::ifstream file = std::ifstream(path.data());
	std::string line;
	std::string text;

	if (!file.is_open())
	{
		const std::string cwdPath = utils::file::get_working_path();
		const std::string filePath = cwdPath + '/' + path.data();
		file = std::ifstream(filePath.data());
		assert(file.is_open());
	}

	if (!file.is_open())
		WARNING("Could not open file: %s", path.data());

	while (std::getline(file, line))
		text += line + "\n";

	return text;
}

static bool exists(const std::string_view &path)
{
	std::ifstream file(path.data());
	return file.is_open();
}

static void write(const std::string_view &path, const std::string_view &contents, bool append)
{
	std::ofstream file;
	file.open(path.data(), (append ? std::ios::app : std::ios::out));
	file.write(contents.data(), contents.size());
	file.close();
}

static void write(const std::string_view &path, const std::vector<char> &contents)
{
	std::ofstream file;
	file.open(path.data(), std::ios::out | std::ios::binary);
	file.write(contents.data(), contents.size());
	file.close();
}
} // namespace rfw::utils::file
