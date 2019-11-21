#pragma once

#include <chrono>
#include <ctime>

namespace rfw::utils
{
using clock = std::chrono::high_resolution_clock;
using time_point = clock::time_point;
using micro_seconds = std::chrono::microseconds;

double getElapsedMinutes();
double getElapsedSeconds();
double getElapsedMilliSeconds();
double getElapsedMicroSeconds();
double getElapsedNanoSeconds();

static time_point getTime() { return clock::now(); }

static double getTimeSinceEpoch() { return std::time(nullptr); }

} // namespace rfw::utils