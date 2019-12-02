#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

layout( location = 1 ) rayPayloadInNV int occluded;

void main()
{
	occluded = 0;
}