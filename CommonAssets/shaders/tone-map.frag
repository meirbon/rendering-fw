#version 330 core

uniform sampler2D tex;

in vec2 uv;

out vec4 Color;

vec3 RRTAndODTFit(vec3 v)
{
	vec3 a = v * (v + 0.0245786f) - 0.000090537f;
	vec3 b = v * (0.983729f * v + 0.4329510f) + 0.238081f;
	return a / b;
}

vec3 ACESFitted(vec3 color)
{
	// sRGB => XYZ => D65_2_D60 => AP1 => RRT_SAT
	const mat3 ACESInputMat = mat3(vec3(0.59719, 0.07600, 0.02840), vec3(0.35458, 0.90834, 0.13383), vec3(0.04823, 0.01566, 0.83777));
	color = ACESInputMat * color;

	// Apply RRT and ODT
	color = RRTAndODTFit(color);

	// ODT_SAT => XYZ => D60_2_D65 => sRGB
	const mat3 ACESOutputMat = mat3(vec3(1.60475, -0.10208, -0.00327), vec3(-0.53108, 1.10813, -0.07276), vec3(-0.07367, -0.00605, 1.07602));
	color = ACESOutputMat * color;

	// Clamp to [0, 1]
	color = clamp(color, 0, 1);

	return color;
}

uniform vec4 params;

void main()
{
	vec2 UV = vec2(uv.x, 1.0 - uv.y);

	vec4 color = texture(tex, UV).rgba;

	vec3 rgb = ACESFitted(max(vec3(0), (color.rgb) - (0.5f * params.x) /* contrastFactor */ + 0.5f + params.y /* brightness */));

	Color = vec4(rgb, color.a);
}