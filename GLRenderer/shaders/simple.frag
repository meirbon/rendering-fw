#version 450

out vec4 Color;

in vec4 Pos;
in vec3 WPos;
in vec3 N;
in vec2 UV;

uniform sampler2D textures[128];

struct Material
{
	uvec4 baseData4;
	uvec4 parameters;
	/* 16 uchars:   x: roughness, metallic, specTrans, specularTint;
					y: diffTrans, anisotropic, sheen, sheenTint;
					z: clearcoat, clearcoatGloss, scatterDistance, relativeIOR;
					w: flatness, ior, dummy1, dummy2. */
	uvec4 t0data4;
	uvec4 t1data4;
	uvec4 t2data4;
	uvec4 n0data4;
	uvec4 n1data4;
	uvec4 n2data4;
	uvec4 sdata4;
	uvec4 rdata4;
	uvec4 m0data4;
	uvec4 m1data4;
#define MAT_FLAGS					(uint(baseData.w))
#define MAT_ISDIELECTRIC			((flags & (1u << 0u)  ) != 0)
#define MAT_DIFFUSEMAPISHDR			((flags & (1u << 1u)  ) != 0)
#define MAT_HASDIFFUSEMAP			((flags & (1u << 2u)  ) != 0)
#define MAT_HASNORMALMAP			((flags & (1u << 3u)  ) != 0)
#define MAT_HASSPECULARITYMAP		((flags & (1u << 4u)  ) != 0)
#define MAT_HASROUGHNESSMAP			((flags & (1u << 5u)  ) != 0)
#define MAT_ISANISOTROPIC			((flags & (1u << 6u)  ) != 0)
#define MAT_HAS2NDNORMALMAP			((flags & (1u << 7u)  ) != 0)
#define MAT_HAS3RDNORMALMAP			((flags & (1u << 8u)  ) != 0)
#define MAT_HAS2NDDIFFUSEMAP		((flags & (1u << 9u)  ) != 0)
#define MAT_HAS3RDDIFFUSEMAP		((flags & (1u << 10u) ) != 0)
#define MAT_HASSMOOTHNORMALS		((flags & (1u << 11u) ) != 0)
#define MAT_HASALPHA				((flags & (1u << 12u) ) != 0)
#define MAT_HASALPHAMAP				((flags & (1u << 13u) ) != 0)
};

layout(std430, binding = 0) buffer materialsBuffer
{
    Material materials[];
};

void main()
{
	Color = vec4(max(vec3(0.2), N), Pos.w);
}