#version 410

out vec4 Color;

in vec4 Pos;
in vec3 WPos;
in vec3 N;

void main()
{
    Color = vec4(max(N, vec3(0.2)), Pos.w);
}