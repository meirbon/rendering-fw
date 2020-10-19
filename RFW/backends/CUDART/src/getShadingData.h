#pragma once

#include <cuda_runtime.h>
#include <glm/glm.hpp>

#ifndef __CUDACC__
#define __CUDACC__
#endif
#include <DeviceStructures.h>
#include <Structures.h>
#include "bsdf/tools.h"

#ifndef MAT_CONSTANTS_H
#define MAT_CONSTANTS_H
__constant__ __device__ rfw::DeviceMaterial *materials;
__constant__ __device__ glm::vec4 *floatTextures;
__constant__ __device__ uint *uintTextures;
#endif

using namespace rfw;
using namespace glm;

inline __device__ glm::vec4 __uchar4_to_float4(const uint v4)
{
	const float r = 1.0f / 256.0f;
	return glm::vec4((float)(v4 & 255u) * r, (float)((v4 >> 8u) & 255u) * r, (float)((v4 >> 16u) & 255u) * r, (float)(v4 >> 24u) * r);
}

inline __device__ glm::vec4 FetchTexel(const glm::vec2 texCoord, const int o, const int w, const int h, const TexelStorage storage = RGBA32)
{
	const float2 tc = make_float2((max(texCoord.x + 1000, 0.0f) * w) - 0.5f, (max(texCoord.y + 1000, 0.0f) * h) - 0.5f);
	const int iu = static_cast<int>(tc.x) % w;
	const int iv = static_cast<int>(tc.y) % h;
#if BILINEAR
	const float fu = tc.x - floor(tc.x);
	const float fv = tc.y - floor(tc.y);
	const float w0 = (1 - fu) * (1 - fv);
	const float w1 = fu * (1 - fv);
	const float w2 = (1 - fu) * fv;
	const float w3 = 1 - (w0 + w1 + w2);
	glm::vec4 p0, p1, p2, p3;
	const uint iu1 = (iu + 1) % w, iv1 = (iv + 1) % h;
	if (storage == RGBA32)
		p0 = __uchar4_to_float4(uintTextures[o + iu + iv * w]), p1 = __uchar4_to_float4(uintTextures[o + iu1 + iv * w]),
		p2 = __uchar4_to_float4(uintTextures[o + iu + iv1 * w]), p3 = __uchar4_to_float4(uintTextures[o + iu1 + iv1 * w]);
	else if (storage == RGBA128)
	{
		p0 = floatTextures[o + iu + iv * w], p1 = floatTextures[o + iu1 + iv * w], p2 = floatTextures[o + iu + iv1 * w], p3 = floatTextures[o + iu1 + iv1 * w];
	}
	return p0 * w0 + p1 * w1 + p2 * w2 + p3 * w3;
#else
	if (storage == RGBA32)
		return __uchar4_to_float4(uintTextures[o + iu + iv * w]);
	else if (storage == RGBA128)
		return floatTextures[o + iu + iv * w];
	else
		return vec4(1.0f);
#endif
}

inline __device__ glm::vec4 FetchTexelTrilinear(const float lambda, const glm::vec2 texCoord, const int offset, const int width, const int height)
{
#if 1
	const int level0 = min(MIPLEVELCOUNT - 1, (int)lambda);
	const int level1 = min(MIPLEVELCOUNT - 1, level0 + 1);
#else
	const int level0 = 0;
	const int level1 = 0;
#endif
	const float f = lambda - floor(lambda);
	// select first MIP level
	uint offset0 = offset;
	uint width0 = width;
	uint height0 = height;
	for (int i = 0; i < level0; i++)
	{
		offset0 += width0 * height0;
		width0 >>= 1u;
		height0 >>= 1u;
	}

	// select second MIP level
	uint offset1 = offset;
	uint width1 = width;
	uint height1 = height;
	for (int i = 0; i < level1; i++)
	{
		offset1 += width1 * height1;
		width1 >>= 1u;
		height1 >>= 1u; // TODO: start at o0, h0, w0
	}

	// read actual data
	const glm::vec4 p0 = FetchTexel(texCoord, offset0, width0, height0);
	const glm::vec4 p1 = FetchTexel(texCoord, offset1, width1, height1);
	// final interpolation
	return (1.0f - f) * p0 + f * p1;
}

inline static __device__ ShadingData getShadingData(const glm::vec3 D, const float u, const float v, const float coneWidth, const rfw::DeviceTriangle &triangle,
													const int instanceIndex, glm::vec3 &N, glm::vec3 &iN, glm::vec3 &T, glm::vec3 &B,
													const glm::mat3 &invTransform)
{
	ShadingData returnValue{};

	const glm::vec4 texu4 = triangle.u4;
	const glm::vec4 texv4 = triangle.v4;

	returnValue.matID = __float_as_uint(texv4.w);
	const auto &mat = reinterpret_cast<const rfw::Material &>(materials[returnValue.matID]);
	const uint flags = mat.flags;
	returnValue.color = mat.getColor();
	returnValue.absorption = mat.getAbsorption();
	returnValue.parameters = mat.parameters;

	const float w = 1.0f - u - v;

	const auto normal0 = triangle.vN0;
	const auto normal1 = triangle.vN1;
	const auto normal2 = triangle.vN2;
	N = glm::vec3(normal0.w, normal1.w, normal2.w);
	iN = N;

	if (Material::hasFlag(flags, HasSmoothNormals))
		iN = normalize(u * normal0 + v * normal1 + w * normal2);

	// Need to normalize since transform can contain scaling
	// TODO: Let rendersystem supply a transform that doesn't contain any scaling
	N = normalize(invTransform * N);
	iN = normalize(invTransform * iN);

	createTangentSpace(iN, T, B);

	// Texturing
	float tu = 0, tv = 0;
	if (Material::hasFlag(flags, HasDiffuseMap) || Material::hasFlag(flags, HasNormalMap) || Material::hasFlag(flags, HasSpecularityMap) ||
		Material::hasFlag(flags, HasRoughnessMap) || Material::hasFlag(flags, Has2ndDiffuseMap) || Material::hasFlag(flags, Has3rdDiffuseMap) ||
		Material::hasFlag(flags, Has2ndNormalMap) || Material::hasFlag(flags, Has3rdNormalMap))
	{
		tu = u * texu4.x + v * texu4.y + w * texu4.z;
		tv = u * texv4.x + v * texv4.y + w * texv4.z;
	}

	if (Material::hasFlag(flags, HasDiffuseMap))
	{
		// determine LOD
		const auto lambda = triangle.getLOD() + log2(coneWidth * (1.0f / abs(dot(-D, N)))); // eq. 26
		// fetch texels
		auto uvscale = glm::vec2(static_cast<float>(mat.uscale0), static_cast<float>(mat.vscale0));
		auto uvoffs = glm::vec2(static_cast<float>(mat.uoffs0), static_cast<float>(mat.voffs0));
		const vec4 texel = FetchTexelTrilinear(lambda, uvscale * (uvoffs + glm::vec2(tu, tv)), mat.texaddr0, static_cast<int>(mat.texwidth0),
											   static_cast<int>(mat.texheight0));

		if (Material::hasFlag(flags, HasAlpha) && texel.w < 0.5f)
		{
			returnValue.flags |= 1;
			return returnValue;
		}

		returnValue.color = returnValue.color * vec3(texel);

		if (Material::hasFlag(flags, Has2ndDiffuseMap)) // must have base texture; second and third layers are additive
		{
			uvscale = glm::vec2(static_cast<float>(mat.uscale1), static_cast<float>(mat.vscale1));
			uvoffs = glm::vec2(static_cast<float>(mat.uoffs1), static_cast<float>(mat.voffs1));
			returnValue.color += vec3(FetchTexelTrilinear(lambda, uvscale * (uvoffs + glm::vec2(tu, tv)), mat.texaddr1, static_cast<int>(mat.texwidth1),
														  static_cast<int>(mat.texheight1)));
		}
		if (Material::hasFlag(flags, Has3rdDiffuseMap))
		{
			uvscale = glm::vec2(static_cast<float>(mat.uscale2), static_cast<float>(mat.vscale2));
			uvoffs = glm::vec2(static_cast<float>(mat.uoffs2), static_cast<float>(mat.voffs2));
			returnValue.color += vec3(FetchTexelTrilinear(lambda, uvscale * (uvoffs + glm::vec2(tu, tv)), mat.texaddr2, static_cast<int>(mat.texwidth2),
														  static_cast<int>(mat.texheight2)));
		}

		// normal mapping
		if (Material::hasFlag(flags, HasNormalMap))
		{
			// fetch bitangent for applying normal map vector to geometric normal
			uvscale = glm::vec2(static_cast<float>(mat.nuscale0), static_cast<float>(mat.nvscale0));
			uvoffs = glm::vec2(static_cast<float>(mat.nuoffs0), static_cast<float>(mat.nvoffs0));
			vec3 shadingNormal =
				(vec3(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), mat.nmapaddr0, mat.nmapwidth0, mat.nmapheight0, RGBA32)) - vec3(0.5f)) * 2.0f;
			if (Material::hasFlag(flags, Has2ndNormalMap))
			{
				uvscale = glm::vec2(static_cast<float>(mat.nuscale1), static_cast<float>(mat.nvscale1));
				uvoffs = glm::vec2(static_cast<float>(mat.nuoffs1), static_cast<float>(mat.nvoffs1));
				vec3 normalLayer1 =
					(vec3(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), mat.nmapaddr1, mat.nmapwidth1, mat.nmapheight1, RGBA32)) - vec3(0.5f)) * 2.0f;
				shadingNormal += normalLayer1;
			}

			if (Material::hasFlag(flags, Has3rdNormalMap))
			{
				uvscale = glm::vec2(static_cast<float>(mat.nuscale1), static_cast<float>(mat.nvscale1));
				uvoffs = glm::vec2(static_cast<float>(mat.nuoffs1), static_cast<float>(mat.nvoffs1));
				vec3 normalLayer2 =
					(vec3(FetchTexel(uvscale * (uvoffs + vec2(tu, tv)), mat.nmapaddr1, mat.nmapwidth1, mat.nmapheight1, RGBA32)) - vec3(0.5f)) * 2.0f;
				shadingNormal += normalLayer2;
			}

			shadingNormal = normalize(shadingNormal);
			iN = normalize(tangentToWorld(shadingNormal, iN, T, B));
		}

		// if (Material::hasFlag(flags, HasAlpha) && texel.w < 0.5f)
		//{
		//	returnValue.flags |= 1;
		//	return returnValue;
		//}

		returnValue.color = returnValue.color * glm::vec3(texel);
	}

	return returnValue;
}