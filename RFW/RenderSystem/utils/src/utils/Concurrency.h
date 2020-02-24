#pragma once

#include "ThreadPool.h"

#include "Logger.h"

#include <algorithm>
#include <numeric>

#include <taskflow/taskflow.hpp>

namespace rfw::utils::concurrency
{
/*
 * Parallel iterator over a range of numbers
 */
template <typename T, typename FUNC> void parallel_for(T first, T last, const FUNC &function)
{
	static tf::Executor executor = tf::Executor();

	// Create task
	tf::Taskflow taskflow = {};
	taskflow.parallel_for((int)first, (int)last, 1, [&function](int i) { function(i); });

	// Run task
	executor.run(taskflow).get();
}
} // namespace rfw::utils::concurrency