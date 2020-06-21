#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

layout( location = 1 ) rayPayloadInEXT int occluded;

void main()
{
	occluded = 0;
}