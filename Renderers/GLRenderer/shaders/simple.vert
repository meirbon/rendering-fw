#version 410

layout(location = 0) in vec4 Vertex;
layout(location = 1) in vec3 Normal;
layout(location = 2) in vec2 TexCoord;

uniform mat4 CamMatrix;
uniform mat3 CamMatrix3x3;

uniform mat4 InstanceMatrices[32];
uniform mat4 InverseMatrices[32];

out vec4 Pos;
out vec3 WPos;
out vec3 N;
out vec2 UV;

void main()
{
	WPos = (InstanceMatrices[gl_InstanceID] * Vertex).xyz;
	Pos = CamMatrix * vec4(WPos, 1.0);
	gl_Position = Pos;
    N = normalize(mat3(InverseMatrices[gl_InstanceID]) * Normal);
	UV = TexCoord;
}