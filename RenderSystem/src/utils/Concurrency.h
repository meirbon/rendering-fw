#pragma once

#include "ThreadPool.h"

#include <utils/Logger.h>

#ifdef _WIN32
#include <ppl.h>
#endif

namespace rfw::utils::concurrency
{

template <typename T, typename FUNC> void parallel_for(T first, T last, const FUNC &function)
{
#if defined(_WIN32) // Microsoft's library is faster than our implementation, use if available
	Concurrency::parallel_for(first, last, function);
	return;
#else
	static ThreadPool loopPool;
	const int poolSize = static_cast<int>(loopPool.size());
	std::vector<std::future<void>> threads(poolSize);

	const auto count = (last - first);
	const auto threadLocalSize = static_cast<int>(ceil(static_cast<float>(count) / poolSize));

	for (int i = 0; i < poolSize; i++)
	{
		threads[i] = loopPool.push([i, &first, &last, &threadLocalSize, &function](int) {
			const int increment = 1;
			const int offset = i * threadLocalSize;
			const int start = first + offset;
			const int end = min(start + threadLocalSize, last);
			for (int j = start; j < end; j += increment)
				function(j);
		});
	}

	for (int i = 0; i < poolSize; i++)
		threads[i].get();
#endif
}
} // namespace rfw::utils::concurrency