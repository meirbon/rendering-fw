#include "Time.h"

using namespace rfw::utils;

static time_point m_startTime = getTime();

double rfw::utils::getElapsedMinutes()
{
	const auto delta = (getTime() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::minutes>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::getElapsedSeconds()
{
	const auto delta = (getTime() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::seconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::getElapsedMilliSeconds()
{
	const auto delta = (getTime() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::milliseconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::getElapsedMicroSeconds()
{
	const auto delta = (getTime() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::microseconds>(delta);
	return static_cast<double>(d.count());
}

double rfw::utils::getElapsedNanoSeconds()
{
	const auto delta = (getTime() - m_startTime);
	const auto d = std::chrono::duration_cast<std::chrono::nanoseconds>(delta);
	return static_cast<double>(d.count());
}