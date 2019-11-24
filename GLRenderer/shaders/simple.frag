#version 410

out vec4 Color;

in vec3 Pos;
in vec3 N;

void main()
{
    Color = vec4(N, 1.0f);
}