#ifndef RANDOM_H
#define RANDOM_H

uint WangHash( uint s )
{
	s = ( s ^ 61 ) ^ ( s >> 16 );
	s *= 9;
	s = s ^ ( s >> 4 );
	s *= 0x27d4eb2d, s = s ^ ( s >> 15 );
	return s;
}

uint RandomInt( inout uint s )
{
	s ^= s << 13;
	s ^= s >> 17;
	s ^= s << 5;
	return s;
}

float RandomFloat( inout uint s ) { return RandomInt( s ) * 2.3283064365387e-10f; }

#endif