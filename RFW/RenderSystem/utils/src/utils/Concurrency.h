#pragma once

#include "ThreadPool.h"

#include "Logger.h"

#ifdef _WIN32
#include <ppl.h>
#endif

#include <algorithm>
#include <numeric>

namespace rfw::utils::concurrency
{
/*
 * Parallel iterator over a range of numbers
 */
template <typename T, typename FUNC> void parallel_for(T first, T last, const FUNC &function)
{
#if defined(_WIN32) // Microsoft's library is faster than our implementation, use if available
	Concurrency::parallel_for(first, last, function);
	return;
#else
	static rfw::utils::ThreadPool pool = {};
	static const int poolSize = pool.size();
	std::vector<std::future<void>> handles(poolSize);

	const int total = static_cast<int>(last) - static_cast<int>(first);
	const int chunk_size = total / poolSize + 1;
	for (int i = 0; i < poolSize; i++)
	{
		handles[i] = pool.push([i, &chunk_size, &total, &function](int) {
			const int start = i * chunk_size;
			const int end = min(start + chunk_size, total);
			for (int j = start; j < end; j++)
			{
				function(j);
			}
		});
	}

	for (auto &handle : handles)
		handle.get();
#endif
}
} // namespace rfw::utils::concurrency