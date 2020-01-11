//
// Created by Mèir Noordermeer on 2019-08-20.
//

#ifndef RENDERING_FW_SRC_UTILS_TIMER_HPP
#define RENDERING_FW_SRC_UTILS_TIMER_HPP

#include <chrono>

namespace rfw
{
namespace utils
{
struct Timer
{
	typedef std::chrono::high_resolution_clock Clock;
	typedef Clock::time_point TimePoint;
	typedef std::chrono::microseconds MicroSeconds;

	TimePoint start;

	inline Timer() : start(get()) {}

	inline float elapsed() const
	{
		auto diff = get() - start;
		auto duration_us = std::chrono::duration_cast<MicroSeconds>(diff);
		return static_cast<float>(duration_us.count()) / 1000.0f;
	}

	static inline TimePoint get() { return Clock::now(); }

	inline void reset() { start = get(); }
};
}; // namespace utils
} // namespace rfw
#endif // RENDERING_FW_SRC_UTILS_TIMER_HPP
