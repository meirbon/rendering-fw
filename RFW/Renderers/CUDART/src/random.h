#pragma once

__host__ __device__ inline unsigned int WangHash(unsigned int s)
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d, s = s ^ (s >> 15);
	return s;
}

__host__ __device__ inline unsigned int RandomInt(unsigned int &s)
{
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

__host__ __device__ inline float RandomFloat(unsigned int &s) { return RandomInt(s) * 2.3283064365387e-10f; }