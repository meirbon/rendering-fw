#include <utils/Timer.h>
#include "SceneMesh.h"

#include "SceneObject.h"

#include "MeshSkin.h"

using namespace glm;

void rfw::SceneMesh::setPose(const rfw::MeshSkin &skin)
{
	using namespace glm;

#if 0
	for (uint i = 0; i < vertexCount; i++)
	{
		const auto idx = i + vertexOffset;

		const uvec4 &j4 = joints.at(i);
		const vec4 &w4 = weights.at(i);

		mat4 skinMatrix = w4.x * skin.jointMatrices.at(j4.x);
		skinMatrix += w4.y * skin.jointMatrices.at(j4.y);
		skinMatrix += w4.z * skin.jointMatrices.at(j4.z);
		skinMatrix += w4.w * skin.jointMatrices.at(j4.w);

		object->vertices.at(idx) = vec4(vec3(object->baseVertices.at(idx)), 1.0f) * skinMatrix;
		object->normals.at(idx) = object->baseNormals.at(idx) * mat3(skinMatrix);
	}

	object->updateTriangles(vertexOffset / 3, vertexCount / 3);
#else
	// https://github.com/jbikker/lighthouse2/blob/master/lib/RenderSystem/host_mesh.cpp
	for (uint s = (vertexCount / 3), t = 0; t < s; t++)
	{
		__m128 tri_vtx[3], tri_nrm[3];
		// adjust vertices of triangle
		for (int t_v = 0; t_v < 3; t_v++)
		{
			// vertex index
			int v = t * 3 + t_v + (vertexOffset / 3);
			// calculate weighted skin matrix
			// skinM = w4.x * skin->jointMat[j4.x]
			//       + w4.y * skin->jointMat[j4.y]
			//       + w4.z * skin->jointMat[j4.z]
			//       + w4.w * skin->jointMat[j4.w];

			// the 4 joint indices
			uvec4 j4 = joints[v];
			// the 4 weights of each joint

			__m128 w4 = _mm_load_ps((const float *)&weights[v]);

			// create scalars for matrix scaling, use same shuffle value to help with uOP cache
			__m256 w4x = _mm256_broadcastss_ps(w4); // w4.x component shuffled to all elements
			w4 = _mm_shuffle_ps(w4, w4, 0b111001);
			__m256 w4y = _mm256_broadcastss_ps(w4); // w4.y component shuffled to all elements
			w4 = _mm_shuffle_ps(w4, w4, 0b111001);
			__m256 w4z = _mm256_broadcastss_ps(w4); // w4.z component shuffled to all elements
			w4 = _mm_shuffle_ps(w4, w4, 0b111001);
			__m256 w4w = _mm256_broadcastss_ps(w4); // w4.w component shuffled to all elements

			// top half of weighted skin matrix
			__m256 skinM_T = _mm256_mul_ps(w4x, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.x])));
			skinM_T = _mm256_fmadd_ps(w4y, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.y])), skinM_T);
			skinM_T = _mm256_fmadd_ps(w4z, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.z])), skinM_T);
			skinM_T = _mm256_fmadd_ps(w4w, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.w])), skinM_T);

			// bottom half of weighted skin matrix
			__m256 skinM_L = _mm256_mul_ps(w4x, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.x]) + 8));
			skinM_L = _mm256_fmadd_ps(w4y, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.y]) + 8), skinM_L);
			skinM_L = _mm256_fmadd_ps(w4z, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.z]) + 8), skinM_L);
			skinM_L = _mm256_fmadd_ps(w4w, _mm256_load_ps(value_ptr(skin.jointMatrices[j4.w]) + 8), skinM_L);

			// double each row so we can do two matrix multiplication at once
			__m256 skinM0 = _mm256_permute2f128_ps(skinM_T, skinM_T, 0x00);
			__m256 skinM1 = _mm256_permute2f128_ps(skinM_T, skinM_T, 0x11);
			__m256 skinM2 = _mm256_permute2f128_ps(skinM_L, skinM_L, 0x00);
			__m256 skinM3 = _mm256_permute2f128_ps(skinM_L, skinM_L, 0x11);

			// load vertices and normal
			__m128 vtxOrig = _mm_load_ps(&object->baseVertices[v].x);
			__m128 normOrig = _mm_maskload_ps(&object->baseNormals[v].x, _mm_set_epi32(0, -1, -1, -1));

			// combine vectors to use AVX2 instead of SSE
			__m256 combined = _mm256_set_m128(normOrig, vtxOrig);

			// multiply vertex with skin matrix, multiply normal with skin matrix
			// using HADD and MUL is faster than OR and DP
			combined = _mm256_hadd_ps(_mm256_hadd_ps(_mm256_mul_ps(combined, skinM0), _mm256_mul_ps(combined, skinM1)),
									  _mm256_hadd_ps(_mm256_mul_ps(combined, skinM2), _mm256_mul_ps(combined, skinM3)));
			// extract vertex and normal from combined vector
			__m128 vtx = _mm256_castps256_ps128(combined);
			__m128 norm = _mm256_extractf128_ps(combined, 1);

			// normalize normal
			norm = _mm_mul_ps(norm, _mm_rsqrt_ps(_mm_dp_ps(norm, norm, 0x77)));

			// store for reuse
			tri_vtx[t_v] = vtx;
			_mm_store_ps(&object->vertices[v].x, vtx);
			tri_nrm[t_v] = norm;
			_mm_maskstore_ps(&object->normals[v].x, _mm_set_epi32(0, -1, -1, -1), norm);
		}

		// get vectors to calculate triangle normal
		__m128 N_a = _mm_sub_ps(tri_vtx[1], tri_vtx[0]);
		__m128 N_b = _mm_sub_ps(tri_vtx[2], tri_vtx[0]);
		// cross product with four shuffles
		// |a.x|   |b.x|   | a.y * b.z - a.z * b.y |
		// |a.y| X |b.y| = | a.z * b.x - a.x * b.z |
		// |a.z|   |b.z|   | a.x * b.y - a.y * b.x |

		// Can be be done with three shuffles...
		// |a.y|   |b.y|   | a.z * b.x - a.x * b.z |
		// |a.z| X |b.z| = | a.x * b.y - a.y * b.x |
		// |a.x|   |b.x|   | a.y * b.z - a.z * b.y |
		// shuffle(..., 0b010010) = [x, y, z] -> [z, x, y] or [y, z, x] -> [x, y, z]
		__m128 N =
			_mm_fmsub_ps(N_b, _mm_shuffle_ps(N_a, N_a, 0b010010), _mm_mul_ps(N_a, _mm_shuffle_ps(N_b, N_b, 0b010010)));
		// reshuffle to get final result
		N = _mm_shuffle_ps(N, N, 0b010010);
		// normalize cross product
		N = _mm_mul_ps(N, _mm_rsqrt_ps(_mm_dp_ps(N, N, 0x77)));
		// insert into Wth element of tri_nrm (xyzw)
		// 0bxx______ -> element to copy from
		// 0b__xx____ -> element to copy to
		// 0b____0000 -> don't set any values to zero
		tri_nrm[0] = _mm_insert_ps(tri_nrm[0], N, 0b00110000);
		tri_nrm[1] = _mm_insert_ps(tri_nrm[1], N, 0b01110000);
		tri_nrm[2] = _mm_insert_ps(tri_nrm[2], N, 0b10110000);

		const auto to = t + (vertexOffset / 3);

		// we use stores, because we can write multiple times to L1
		_mm_store_ps(&object->triangles[to].vertex0.x, tri_vtx[0]);
		_mm_store_ps(&object->triangles[to].vertex1.x, tri_vtx[1]);
		_mm_store_ps(&object->triangles[to].vertex2.x, tri_vtx[2]);
		// store to [vN0 (float3), Nx (float)]
		_mm_store_ps(&object->triangles[to].vN0.x, tri_nrm[0]);
		// store to [vN1 (float3), Ny (float)]
		_mm_store_ps(&object->triangles[to].vN1.x, tri_nrm[1]);
		// store to [vN1 (float3), Nz (float)]
		_mm_store_ps(&object->triangles[to].vN2.x, tri_nrm[2]);
	}
#endif
	object->dirty = true;
}

void rfw::SceneMesh::setPose(rfw::utils::ArrayProxy<float> weights)
{
	assert(weights.size() == poses.size() - 1);
	const auto weightCount = weights.size();

	for (uint s = vertexCount, i = 0; i < s; i++)
	{
		const auto idx = i + vertexOffset;
		object->vertices.at(idx) = vec4(poses.at(0).positions.at(i), 1.0f);
		object->normals.at(idx) = poses.at(0).normals.at(i);

		for (uint j = 1; j <= weightCount; j++)
		{
			const auto &pose = poses.at(j);

			object->vertices.at(idx) += weights.at(j - 1) * vec4(pose.positions.at(i), 0);
			object->normals.at(idx) += weights.at(j - 1) * pose.normals.at(i);
		}
	}

	object->updateTriangles(vertexOffset / 3, vertexCount / 3);
	object->dirty = true;
}

void rfw::SceneMesh::setTransform(const glm::mat4 &transform)
{
	vec4 *baseVertex = &object->vertices.at(vertexOffset);
	vec3 *baseNormal = &object->normals.at(vertexOffset);

	const auto matrix3x3 = mat3(transform);

	for (uint i = 0, idx = vertexOffset; i < vertexCount; i++, idx++)
	{
		baseVertex[i] = transform * object->baseVertices.at(idx);
		baseNormal[i] = matrix3x3 * object->baseNormals.at(idx);
	}

	const auto offset = vertexOffset / 3;
	for (uint s = (vertexCount / 3), i = 0, triIdx = offset; i < s; i++, triIdx++)
	{
		const auto idx = i * 3;
		auto &tri = object->triangles.at(triIdx);
		tri.vertex0 = vec3(baseVertex[idx + 0]);
		tri.vertex1 = vec3(baseVertex[idx + 1]);
		tri.vertex2 = vec3(baseVertex[idx + 2]);

		tri.vN0 = baseNormal[idx + 0];
		tri.vN1 = baseNormal[idx + 1];
		tri.vN2 = baseNormal[idx + 2];
	}

	object->dirty = true;
}
