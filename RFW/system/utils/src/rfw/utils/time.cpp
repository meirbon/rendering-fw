#include "Time.h"

using namespace rfw::utils;

static time_point m_startTime = get_time();

double rfw::utils::get_elapsed_minutes()
{
	const auto delta = (get_time() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::minutes>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::get_elapsed_seconds()
{
	const auto delta = (get_time() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::seconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::get_elapsed_milli_seconds()
{
	const auto delta = (get_time() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::get_elapsed_micro_seconds()
{
	const auto delta = (get_time() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::microseconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::get_elapsed_nano_seconds()
{
	const auto delta = (get_time() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
	return static_cast<double>(d.count());
}