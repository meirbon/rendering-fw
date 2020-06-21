#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_EXT_ray_tracing : require

layout( location = 0 ) rayPayloadInEXT vec4 hitData;

hitAttributeEXT vec2 attribs;

void main()
{
	const uint bary = uint(65535.0f * attribs.x) + (uint(65535.0f * attribs.y) << 16);
	hitData = vec4(uintBitsToFloat(bary), uintBitsToFloat(gl_InstanceCustomIndexEXT), intBitsToFloat(int(gl_PrimitiveID)), gl_RayTmaxEXT);
}