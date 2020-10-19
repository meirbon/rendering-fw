#version 330 core

uniform sampler2D tex;

in vec2 uv;

out vec4 Color;

void main()
{
	Color = vec4(texture(tex, uv).rgb, 1.0 );
}