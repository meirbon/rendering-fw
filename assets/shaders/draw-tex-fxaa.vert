#version 330 core

layout (location = 0) in vec3 Pos;
layout (location = 1) in vec2 UV;

uniform mat4 view;
uniform float rt_w;
uniform float rt_h;

out vec2 uv;
out vec4 posPos;

void main()
{
    float FXAA_SUBPIX_SHIFT = 1.0 / 4.0;

    uv = UV;
    gl_Position = view * vec4(Pos, 1.0);

    vec2 rcpFrame = vec2(rt_w, rt_h);
    posPos.xy = UV.xy;
    posPos.zw = UV.xy - (rcpFrame * (0.5 + FXAA_SUBPIX_SHIFT));
}