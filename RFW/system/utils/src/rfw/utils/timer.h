#pragma once

#include <chrono>

namespace rfw::utils
{
struct timer
{
	using clock = std::chrono::high_resolution_clock;
	using time_point = clock::time_point;
	using micro_seconds = std::chrono::microseconds;

	time_point start;

	inline timer() : start(get()) {}

	inline float elapsed() const
	{
		auto diff = get() - start;
		auto duration_us = std::chrono::duration_cast<micro_seconds>(diff);
		return static_cast<float>(duration_us.count()) / 1000.0f;
	}

	static inline time_point get() { return clock::now(); }

	inline void reset() { start = get(); }
};
} // namespace rfw::utils
