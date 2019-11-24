#version 410

layout(location = 0) in vec4 Vertex;
layout(location = 1) in vec3 Normal;

uniform mat4 CamMatrix;

out vec3 Pos;
out vec3 N;

void main()
{
    vec4 pos = CamMatrix * Vertex;
    Pos = pos.xyz;
    N = Normal;
}