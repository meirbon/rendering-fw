#version 410

layout(location = 0) in vec4 Vertex;
layout(location = 1) in vec3 Normal;

uniform mat4 CamMatrix;
uniform mat3 CamMatrix3x3;

uniform mat4 InstanceMatrices[32];
uniform mat4 InverseMatrices[32];

out vec4 Pos;
out vec3 WPos;
out vec3 N;

void main()
{
	WPos = (InstanceMatrices[gl_InstanceID] * Vertex).xyz;
	Pos = CamMatrix * vec4(WPos, 1.0);
	gl_Position = Pos;
    N = (InstanceMatrices[gl_InstanceID] * vec4(Normal, 0)).xyz;
}