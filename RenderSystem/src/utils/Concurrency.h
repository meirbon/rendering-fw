#pragma once

#include "ThreadPool.h"

#include <utils/Logger.h>

#ifdef _WIN32
#include <ppl.h>
#endif

#include <algorithm>
#include <execution>
#include <numeric>

namespace rfw::utils::concurrency
{
/*
 * Parallel iterator over a range of numbers
 */
template <typename T, typename FUNC> void parallel_for(T first, T last, const FUNC &function)
{
#ifdef _WIN32 // Microsoft's library is faster than our implementation, use if available
	Concurrency::parallel_for(first, last, function);
	return;
#endif

	std::vector<int> r(last - first);
	std::iota(std::begin(r), std::end(r), first);
	std::for_each_n(std::execution::par_unseq, r.begin(), last, [&function](T &item) { function(item); });
}
} // namespace rfw::utils::concurrency