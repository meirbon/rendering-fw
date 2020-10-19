#pragma once
#include "rng.h"

#include <random>

namespace rfw::utils
{

class mersenne_twister : public rfw::utils::rng
{
  public:
	mersenne_twister() : mt_gen(std::random_device()()) {}

	float rand(float range) override final { return rand_uint() * 2.3283064365387e-10f * range; }

	unsigned int rand_uint() override final { return mt_gen(); }

  private:
	std::mt19937 mt_gen;
};
} // namespace rfw::utils