#pragma once

#include "RandomGenerator.h"

#include <random>

namespace rfw::utils
{

class Xor128 : public rfw::utils::RandomGenerator
{
  public:
	Xor128() = default;
	Xor128(unsigned int seed)
	{
		std::random_device mt_rd;
		x = mt_rd() * seed;
	}

	unsigned int RandomUint() override final
	{
		uint t;
		t = x ^ (x << 11);
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