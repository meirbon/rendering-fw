#pragma once

#include <cuda_runtime.h>

#include <utils/Logger.h>

static void _CheckCUDA(const char *file, int line, cudaError code, bool abort = true)
{
	if (code != cudaSuccess)
	{
		char buffer[1024];
		sprintf(buffer, "%s :: CUDA Error on line %i : %s", file, line, cudaGetErrorString(code));

		if (abort)
			rfw::utils::logger::err(buffer);
		else
			rfw::utils::logger::warning(buffer);
	}
}

#define CheckCUDA(c) _CheckCUDA(__FILE__, __LINE__, c)