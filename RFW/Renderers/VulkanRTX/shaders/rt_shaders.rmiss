#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

#include "../src/Bindings.h"
#include "tools.glsl"

layout( location = 0 ) rayPayloadInEXT vec4 hitData;

void main()
{
	hitData = vec4(0.0f, uintBitsToFloat(0), intBitsToFloat(-1), -1.0f);
}