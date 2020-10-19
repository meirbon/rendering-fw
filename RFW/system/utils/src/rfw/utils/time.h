#pragma once

#include <chrono>
#include <ctime>

namespace rfw::utils
{
using clock = std::chrono::high_resolution_clock;
using time_point = clock::time_point;
using micro_seconds = std::chrono::microseconds;

double get_elapsed_minutes();
double get_elapsed_seconds();
double get_elapsed_milli_seconds();
double get_elapsed_micro_seconds();
double get_elapsed_nano_seconds();

static time_point get_time() { return clock::now(); }

static double get_time_since_epoch() { return static_cast<double>(std::time(nullptr)); }

} // namespace rfw::utils