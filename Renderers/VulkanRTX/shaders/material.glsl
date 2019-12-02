#include "structures.glsl"
#include "tools.glsl"
#include "sampling.glsl"

layout(set = 0, binding = cMATERIALS) buffer materialBuffer { DeviceMaterial data[]; }
materials;

void GetShadingData(const vec3 D,				  // IN: incoming ray direction
					const float u, const float v, // barycentrics
					const float coneWidth,		  // ray cone width, for texture LOD
					const DeviceTriangle tri,	 // triangle data
					inout ShadingData retVal,	 // material properties for current intersection
					inout vec3 N, inout vec3 iN,  // geometric normal, interpolated normal
					inout vec3 T, inout vec3 B,   // tangent vector
					const mat3 invTransform		  // inverse instance transformation matrix
)
{
	const float w = 1.0f - u - v;
	const vec4 N0 = tri.vN0;
	const vec4 N1 = tri.vN1;
	const vec4 N2 = tri.vN2;
	const vec4 V4 = tri.v4;
	const vec4 T4 = tri.T;

	const DeviceMaterial mat = materials.data[floatBitsToUint(tri.v4.w)]; // material data

	const uvec4 baseData = mat.baseData4;

	const vec2 base_rg = unpackHalf2x16(baseData.x);
	const vec2 base_b_medium_r = unpackHalf2x16(baseData.y);
	const vec2 medium_gb = unpackHalf2x16(baseData.z);
	const uint flags = MAT_FLAGS;

	retVal.color = vec4(base_rg.x, base_rg.y, base_b_medium_r.x, 0);			 // uint flags;
	retVal.absorption = vec4(base_b_medium_r.y, medium_gb.x, medium_gb.y, T4.w); // uint matID;
	retVal.parameters = mat.parameters;

	N = vec3(N0.w, N1.w, N2.w);
	iN = normalize(w * N0.xyz + u * N1.xyz + v * N2.xyz);

	// Transform normals from local space to world space
	N = invTransform * N;
	iN = invTransform * iN;

	// Calculating tangent space is faster than loading from memory
	createTangentSpace(iN, T, B);

	// Texturing
	float tu, tv;
	if (MAT_HASDIFFUSEMAP || MAT_HAS2NDDIFFUSEMAP || MAT_HAS3RDDIFFUSEMAP || MAT_HASSPECULARITYMAP || MAT_HASNORMALMAP || MAT_HAS2NDNORMALMAP ||
		MAT_HAS3RDNORMALMAP || MAT_HASROUGHNESSMAP)
	{
		const vec4 tdata0 = tri.u4;
		tu = w * tri.u4.x + u * tri.u4.y + v * tri.u4.z;
		tv = w * tri.v4.x + u * tri.v4.y + v * tri.v4.z;
	}

	if (MAT_HASDIFFUSEMAP)
	{
		// Determine LOD
		const float lambda = tri.B.w + log2(coneWidth * (1.0 / abs(dot(D, N))));
		uvec4 data = mat.t0data4;

		vec2 uvscale = unpackHalf2x16(data.y);
		vec2 uvoffs = unpackHalf2x16(data.z);

		// Fetch texels
		const vec4 texel = FetchTexelTrilinear(lambda, uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u));
		if (MAT_HASALPHA && texel.w < 0.5f)
		{
			retVal.color.w = uintBitsToFloat(1);
			return;
		}
		retVal.color.xyz = retVal.color.xyz * texel.xyz;
		if (MAT_HAS2NDDIFFUSEMAP) // must have base texture; second and third layers are additive
		{
			data = mat.t1data4;
			uvscale = unpackHalf2x16(data.y);
			uvoffs = unpackHalf2x16(data.z);
			retVal.color.xyz += FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u), ARGB32).xyz;
		}
		if (MAT_HAS3RDDIFFUSEMAP)
		{
			data = mat.t2data4;
			uvscale = unpackHalf2x16(data.y);
			uvoffs = unpackHalf2x16(data.z);
			retVal.color.xyz += FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u), ARGB32).xyz;
		}
	}
	// Normal mapping
	if (MAT_HASNORMALMAP)
	{
		uvec4 data = mat.n0data4;
		vec2 uvscale = unpackHalf2x16(data.y);
		vec2 uvoffs = unpackHalf2x16(data.z);
		vec3 shadingNormal =
			(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u), ARGB32).xyz - vec3(0.5f)) * 2.0f;

		if (MAT_HAS2NDNORMALMAP)
		{
			data = mat.n1data4;
			const vec2 uvscale = unpackHalf2x16(data.y);
			const vec2 uvoffs = unpackHalf2x16(data.z);
			const vec3 normalLayer1 =
				(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16), ARGB32).xyz - vec3(0.5f)) * 2.0f;
			shadingNormal += normalLayer1;
		}

		if (MAT_HAS3RDNORMALMAP)
		{
			data = mat.n2data4;
			const vec2 uvscale = unpackHalf2x16(data.y);
			const vec2 uvoffs = unpackHalf2x16(data.z);
			const vec3 normalLayer2 =
				(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u), ARGB32).xyz - vec3(0.5f)) * 2.0f;
			shadingNormal += normalLayer2;
		}

		shadingNormal = normalize(shadingNormal);
		iN = normalize(shadingNormal * T + shadingNormal.y * B + shadingNormal.z * iN);
	}

	if (MAT_HASROUGHNESSMAP)
	{
		const uvec4 data = mat.rdata4;
		const vec2 uvscale = unpackHalf2x16(data.y);
		const vec2 uvoffs = unpackHalf2x16(data.z);
		const uint blend =
			(retVal.parameters.x & 0xffffff) +
			(uint(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), int(data.w), int(data.x & 0xFFFFu), int(data.x >> 16u), ARGB32).x * 255.0f) << 24u);
		retVal.parameters.x = blend;
	}
}