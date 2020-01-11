#version 460
#extension GL_GOOGLE_include_directive : require
#extension GL_NV_ray_tracing : require

layout( location = 0 ) rayPayloadInNV vec4 hitData;

hitAttributeNV vec2 attribs;

void main()
{
	const uint bary = uint(65535.0f * attribs.x) + (uint(65535.0f * attribs.y) << 16);
	hitData = vec4(uintBitsToFloat(bary), uintBitsToFloat(gl_InstanceCustomIndexNV), intBitsToFloat(int(gl_PrimitiveID)), gl_RayTmaxNV);
}