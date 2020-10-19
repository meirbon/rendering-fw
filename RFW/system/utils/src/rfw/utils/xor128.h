#pragma once

#include "rng.h"

#include <random>

namespace rfw::utils
{

class xor128 : public rfw::utils::rng
{
  public:
	xor128() = default;
	xor128(unsigned int seed)
	{
		std::random_device mt_rd;
		x = mt_rd() * (seed + 1);
	}

	unsigned int rand_uint() override final
	{
		unsigned int t = x ^ (x << 11);
		x = y;
		y = z;
		z = w;
		return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
	}

  private:
	unsigned int x = 123456789;
	unsigned int y = 362436069;
	unsigned int z = 521288629;
	unsigned int w = 88675123;
};
} // namespace rfw::utils