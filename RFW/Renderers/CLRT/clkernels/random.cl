#ifndef RANDOM_H
#define RANDOM_H

uint TauStep(int s1, int s2, int s3, uint M, uint *seed);
uint HQIRand(uint *seed);
uint SeedRandom(uint s);
uint RandomInt(uint *seed);
float RandomFloat(uint *seed);
uint WangHash(unsigned int s);

uint TauStep(int s1, int s2, int s3, uint M, uint *seed)
{
	uint b = (((*seed << s1) ^ *seed) >> s2);
	*seed = (((*seed & M) << s3) ^ b);
	return *seed;
}

uint HQIRand(uint *seed)
{
	uint z1 = TauStep(13, 19, 12, 429496729, seed);
	uint z2 = TauStep(2, 25, 4, 4294967288, seed);
	uint z3 = TauStep(3, 11, 17, 429496280, seed);
	uint z4 = 1664525 * *seed + 1013904223;
	return z1 ^ z2 ^ z3 ^ z4;
}

uint SeedRandom(uint s)
{
	uint seed = s * 1099087573;
	seed = HQIRand(&seed);
	return seed;
}

uint RandomInt(uint *seed)
{
	// Marsaglia Xor32; see http://excamera.com/sphinx/article-xorshift.html
	// also see https://github.com/WebDrake/xorshift/blob/master/xorshift.c for higher quality variants
	*seed ^= *seed << 13;
	*seed ^= *seed >> 17;
	*seed ^= *seed << 5;
	return *seed;
}

float RandomFloat(uint *seed)
{
	return 0.0f;
#if 0
	return RandomInt(seed) * 2.3283064365387e-10f;
#endif
}

unsigned int WangHash(uint s)
{
	s = (s ^ 61) ^ (s >> 16);
	s *= 9;
	s = s ^ (s >> 4);
	s *= 0x27d4eb2d, s = s ^ (s >> 15);
	return s;
}

float blueNoiseSampler(global uint *blueNoise, int x, int y, int sampleIdx, int sampleDimension)
{
	// wrap arguments
	x &= 127;
	y &= 127;
	sampleIdx &= 255;
	sampleDimension &= 255;

	// xor index based on optimized ranking
	int rankedSampleIndex = sampleIdx ^ blueNoise[sampleDimension + (x + y * 128) * 8 + 65536 * 3];

	// fetch value in sequence
	int value = blueNoise[sampleDimension + rankedSampleIndex * 256];

	// if the dimension is optimized, xor sequence value based on optimized scrambling
	value ^= blueNoise[(sampleDimension & 7) + (x + y * 128) * 8 + 65536];

	// convert to float and return
	return (0.5f + value) * (1.0f / 256.0f);
}

#endif