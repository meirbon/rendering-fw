#version 330 core

uniform sampler2D tex;
in vec2 uv;
in vec4 posPos;
out vec4 Color;

// http://www.geeks3d.com/20110405/fxaa-fast-approximate-anti-aliasing-demo-glsl-opengl-test-radeon-geforce/3/
uniform float rt_w;
uniform float rt_h;
uniform float FXAA_SPAN_MAX = 8.0;
uniform float FXAA_REDUCE_MUL = 1.0 / 8.0;

// Output of FxaaVertexShader interpolated across screen.
// Input texture.
// Constant {1.0/frameWidth, 1.0/frameHeight}.
vec3 FxaaPixelShader(vec4 posPos, sampler2D tex, vec2 rcpFrame)
{
    /*---------------------------------------------------------*/
    #define FXAA_REDUCE_MIN   (1.0 / 64.0)
    /*---------------------------------------------------------*/

    vec3 rgbNW = textureLod(tex, posPos.zw, 0.0).xyz;
    vec3 rgbNE = textureLodOffset(tex, posPos.zw, 0.0, ivec2(1, 0)).xyz;
    vec3 rgbSW = textureLodOffset(tex, posPos.zw, 0.0, ivec2(0, 1)).xyz;
    vec3 rgbSE = textureLodOffset(tex, posPos.zw, 0.0, ivec2(1, 1)).xyz;
    vec3 rgbM  = textureLod(tex, posPos.xy, 0.0).xyz;

    vec3 luma = vec3(0.299, 0.587, 0.114);
    float lumaNW = dot(rgbNW, luma);
    float lumaNE = dot(rgbNE, luma);
    float lumaSW = dot(rgbSW, luma);
    float lumaSE = dot(rgbSE, luma);
    float lumaM  = dot(rgbM, luma);

    float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
    float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

    vec2 dir;
    dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
    dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

    float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 * FXAA_REDUCE_MUL), FXAA_REDUCE_MIN);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = min(vec2(FXAA_SPAN_MAX, FXAA_SPAN_MAX), max(vec2(-FXAA_SPAN_MAX, -FXAA_SPAN_MAX), dir * rcpDirMin)) * rcpFrame.xy;

    vec3 rgbA = 0.5 * (textureLod(tex, posPos.xy + dir * (1.0/3.0 - 0.5), 0.0).xyz + textureLod(tex, posPos.xy + dir * (2.0/3.0 - 0.5), 0.0).xyz);
    vec3 rgbB = rgbA * 0.5 + (1.0/4.0) * (textureLod(tex, posPos.xy + dir * (0.0/3.0 - 0.5), 0.0).xyz + textureLod(tex, posPos.xy + dir * (3.0/3.0 - 0.5), 0.0).xyz);
    float lumaB = dot(rgbB, luma);

    if ((lumaB < lumaMin) || (lumaB > lumaMax))
    {
        return rgbA;
    }

    return rgbB;
}

vec4 PostFX(sampler2D tex, vec2 uv, float time)
{
    vec2 rcpFrame = vec2(rt_w, rt_h);
    return vec4(FxaaPixelShader(posPos, tex, rcpFrame), 1.0);
}

void main()
{
    Color = PostFX(tex, uv, 0.0);
    //	Color = vec4(texture(tex, uv).rgb, 1.0);
}