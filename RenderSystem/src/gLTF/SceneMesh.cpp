#include <utils/Timer.h>
#include "SceneMesh.h"

#include "SceneObject.h"

#include "Skinning.h"

#define ALLOW_INDEXED_DATA 1
#define ALLOW_INDEXED_ANIM_DATA 1

using namespace glm;

rfw::SceneMesh::SceneMesh() { flags |= INITIAL_PRIM; }
rfw::SceneMesh::SceneMesh(const rfw::SceneObject &obj) : object(const_cast<rfw::SceneObject *>(&obj)) { flags |= INITIAL_PRIM; }

void rfw::SceneMesh::setPose(const rfw::MeshSkin &skin)
{
	using namespace glm;

	auto vertices = getVertices();
	const auto baseVertices = getBaseVertices();
	auto normals = getNormals();
	const auto baseNormals = getBaseNormals();

	for (int i = 0, s = static_cast<int>(vertexCount); i < s; i++)
	{
#if ROW_MAJOR_MESH_SKIN
		// the 4 joint indices
		const uvec4 &j4 = joints[i];
		// the 4 weights of each joint
		__m128 w4 = _mm_load_ps(value_ptr(weights[i]));

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
		const __m256 skinM0 = _mm256_permute2f128_ps(skinM_T, skinM_T, 0x00);
		const __m256 skinM1 = _mm256_permute2f128_ps(skinM_T, skinM_T, 0x11);
		const __m256 skinM2 = _mm256_permute2f128_ps(skinM_L, skinM_L, 0x00);
		const __m256 skinM3 = _mm256_permute2f128_ps(skinM_L, skinM_L, 0x11);

		// load vertices and normal
		__m128 vtxOrig = _mm_load_ps(value_ptr(object->baseVertices[i]));
		__m128 normOrig = _mm_maskload_ps(value_ptr(object->baseNormals[i]), _mm_set_epi32(0, ~0, ~0, ~0));

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
		_mm_store_ps(value_ptr(object->vertices[i]), vtx);
		_mm_maskstore_ps(value_ptr(object->normals[i]), _mm_set_epi32(0, -1, -1, -1), norm);
#else
		const uvec4 &j4 = joints.at(i);
		const vec4 &w4 = weights.at(i);

		const mat4 mx = skin.jointMatrices.at(j4.x) * w4.x;
		const mat4 my = skin.jointMatrices.at(j4.y) * w4.y;
		const mat4 mz = skin.jointMatrices.at(j4.z) * w4.z;
		const mat4 mw = skin.jointMatrices.at(j4.w) * w4.w;
		const mat4 skinMatrix = mx + my + mz + mw;

		vertices[i] = skinMatrix * baseVertices[i];
		normals[i] = skinMatrix * vec4(baseNormals[i], 0);
#endif
	}

	if (flags & HAS_INDICES)
		object->updateTriangles(faceOffset, faceOffset + faceCount);
	else
		object->updateTriangles(vertexOffset / 3, vertexOffset / 3 + vertexCount / 3);
	object->dirty = true;
}

void rfw::SceneMesh::setPose(const std::vector<float> &wghts)
{
	assert(wghts.size() == poses.size() - 1);
	const auto weightCount = wghts.size();

	for (int s = static_cast<int>(vertexCount), i = 0; i < s; i++)
	{
		const auto idx = i + vertexOffset;
		object->vertices.at(idx) = vec4(poses.at(0).positions.at(i), 1.0f);
		object->normals.at(idx) = poses.at(0).normals.at(i);

		for (int j = 1; j <= weightCount; j++)
		{
			const auto &pose = poses.at(j);

			object->vertices.at(idx) += wghts.at(j - 1) * vec4(pose.positions.at(i), 0);
			object->normals.at(idx) += wghts.at(j - 1) * pose.normals.at(i);
		}

		object->normals.at(idx).y *= -1.0f;
		object->normals.at(idx).z *= -1.0f;
	}

	object->updateTriangles(vertexOffset / 3, vertexCount / 3);
	object->dirty = true;
}

void rfw::SceneMesh::setTransform(const glm::mat4 &transform)
{
	const auto matrix3x3 = mat3(transform);

	auto vertices = getVertices();
	const auto baseVertices = getBaseVertices();
	auto normals = getNormals();
	const auto baseNormals = getBaseNormals();

	for (int i = 0; i < vertexCount; i++)
	{
		vertices[i] = transform * baseVertices[i];
		normals[i] = matrix3x3 * baseNormals[i];
	}

	if (flags & HAS_INDICES)
	{
		const auto offset = vertexOffset / 3;
		for (uint s = (vertexCount / 3), i = 0, triIdx = offset; i < s; i++, triIdx++)
		{
			const auto idx = i * 3;
			auto &tri = object->triangles.at(triIdx);
			tri.vertex0 = vec3(vertices[idx + 0]);
			tri.vertex1 = vec3(vertices[idx + 1]);
			tri.vertex2 = vec3(vertices[idx + 2]);

			const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));

			tri.vN0 = normals[idx + 0];
			tri.vN1 = normals[idx + 1];
			tri.vN2 = normals[idx + 2];

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;
		}
	}
	else
	{
		const auto offset = vertexOffset / 3;
		for (uint s = (vertexCount / 3), i = 0, triIdx = offset; i < s; i++, triIdx++)
		{
			const auto idx = i * 3;
			auto &tri = object->triangles.at(triIdx);
			tri.vertex0 = vec3(vertices[idx + 0]);
			tri.vertex1 = vec3(vertices[idx + 1]);
			tri.vertex2 = vec3(vertices[idx + 2]);

			const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));

			tri.vN0 = normals[idx + 0];
			tri.vN1 = normals[idx + 1];
			tri.vN2 = normals[idx + 2];

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;
		}
	}

	object->dirty = true;
}

vec4 *rfw::SceneMesh::getVertices() { return &object->vertices.at(vertexOffset); }
const vec4 *rfw::SceneMesh::getVertices() const { return &object->vertices.at(vertexOffset); }

vec4 *rfw::SceneMesh::getBaseVertices() { return &object->baseVertices.at(vertexOffset); }
const vec4 *rfw::SceneMesh::getBaseVertices() const { return &object->baseVertices.at(vertexOffset); }

glm::vec3 *rfw::SceneMesh::getNormals() { return &object->normals.at(vertexOffset); }
const glm::vec3 *rfw::SceneMesh::getNormals() const { return &object->normals.at(vertexOffset); }

glm::vec3 *rfw::SceneMesh::getBaseNormals() { return &object->baseNormals.at(vertexOffset); }
const glm::vec3 *rfw::SceneMesh::getBaseNormals() const { return &object->baseNormals.at(vertexOffset); }

void rfw::SceneMesh::addPrimitive(const std::vector<int> &indces, const std::vector<glm::vec3> &verts, const std::vector<glm::vec3> &nrmls,
								  const std::vector<glm::vec2> &uvs, const std::vector<rfw::SceneMesh::Pose> &pses, const std::vector<glm::uvec4> &jnts,
								  const std::vector<glm::vec4> &wghts, const int materialIdx)
{
	std::vector<int> indices = indces;

	if (!jnts.empty() && (object->flags & SceneObject::ALLOW_INDICES))
	{
		if (flags | INITIAL_PRIM)
		{
			flags &= ~INITIAL_PRIM;
			if (!indices.empty()) // This mesh will be indexed
			{
				flags |= HAS_INDICES;
				faceOffset = object->indices.size();
			}

			vertexOffset = object->baseVertices.size();
		}

		vertexCount += verts.size();

		if (flags & HAS_INDICES)
		{
			if (indices.empty()) // Generate indices if current mesh already consists of indexed data
			{
				for (int i = 0; i < verts.size(); i++)
					indices.push_back(i);
			}

			faceCount += indices.size() / 3;
			object->triangles.resize(object->triangles.size() + faceCount);
			object->materialIndices.resize(object->materialIndices.size() + faceCount, materialIdx);
			object->indices.reserve(object->indices.size() + faceCount);

			object->baseVertices.reserve(object->baseVertices.size() + indices.size());
			object->baseNormals.reserve(object->baseNormals.size() + indices.size());
			object->texCoords.reserve(object->texCoords.size() + indices.size());

			for (int i = 0; i < indices.size(); i += 3)
			{
				const uvec3 index = uvec3(indices[i], indices[i + 1], indices[i + 2]) + vertexOffset;
				object->indices.emplace_back(index);
			}
		}
		else
		{
			faceCount += verts.size() / 3;
			object->materialIndices.resize(object->materialIndices.size() + (verts.size() / 3), materialIdx);
			object->triangles.resize(object->triangles.size() + (verts.size() / 3));

			object->baseVertices.reserve(object->baseVertices.size() + verts.size());
			object->baseNormals.reserve(object->baseNormals.size() + verts.size());
			object->texCoords.reserve(object->texCoords.size() + verts.size());
		}

		poses.resize(pses.size());
		for (size_t i = 0; i < poses.size(); i++)
		{
			const auto &origPose = pses.at(i);
			auto &pose = poses.at(i);

			pose.positions = origPose.positions;
			pose.normals = origPose.normals;
		}

		if (!jnts.empty())
		{
			assert(!wghts.empty());
			assert(jnts.size() == wghts.size());

			if (flags & HAS_INDICES)
			{
				joints.reserve(indices.size());
				weights.reserve(indices.size());

				for (size_t s = indices.size(), i = 0; i < s; i++)
				{
					const auto idx = indices.at(i);

					joints.push_back(jnts.at(idx));
					weights.push_back(wghts.at(idx));
				}
			}
			else
			{
				joints.reserve(verts.size());
				weights.reserve(verts.size());

				for (size_t s = verts.size(), i = 0; i < s; i++)
				{
					joints.push_back(jnts.at(i));
					weights.push_back(wghts.at(i));
				}
			}
		}

		for (int i = 0, s = static_cast<int>(verts.size()); i < s; i++)
		{
			object->baseVertices.emplace_back(verts.at(i), 1);
			object->baseNormals.push_back(nrmls.at(i));
			if (!uvs.empty())
				object->texCoords.push_back(uvs.at(i));
			else
				object->texCoords.emplace_back(0.0f);
		}
	}
	else if (!indices.empty())
	{
		if (flags | INITIAL_PRIM)
			vertexOffset = object->baseVertices.size();

		faceCount += indices.size() / 3;
		vertexCount += indices.size();

		// Allocate data
		const auto triangleOffset = object->triangles.size();

		object->materialIndices.resize(object->materialIndices.size() + (indices.size() / 3), materialIdx);
		object->triangles.resize(object->triangles.size() + (indices.size() / 3));

		object->baseVertices.reserve(object->baseVertices.size() + indices.size());
		object->baseNormals.reserve(object->baseNormals.size() + indices.size());
		object->texCoords.reserve(object->texCoords.size() + indices.size());

		poses.resize(poses.size());
		for (size_t i = 0; i < poses.size(); i++)
		{
			const auto &origPose = pses.at(i);
			auto &pose = poses.at(i);

			pose.positions.reserve(indices.size());
			pose.normals.reserve(indices.size());

			for (int idx : indices)
			{
				pose.positions.push_back(origPose.positions.at(idx));
				pose.normals.emplace_back(origPose.normals.at(idx));
			}
		}

		if (!jnts.empty())
		{
			joints.reserve(indices.size());
			weights.reserve(indices.size());

			for (int s = static_cast<int>(indices.size()), i = 0; i < s; i++)
			{
				const auto idx = indices.at(i);

				joints.push_back(jnts.at(idx));
				weights.push_back(wghts.at(idx));
			}
		}

		// Add per-vertex data
		for (int s = static_cast<int>(indices.size()), i = 0; i < s; i++)
		{
			const auto idx = indices.at(i);

			object->baseVertices.emplace_back(verts.at(idx), 1.0f);
			object->baseNormals.push_back(nrmls.at(idx));
			if (!uvs.empty())
				object->texCoords.push_back(uvs.at(idx));
			else
				object->texCoords.emplace_back(0.0f);
		}
	}
	else
	{
		if (flags | INITIAL_PRIM)
			vertexOffset = object->baseVertices.size();

		faceCount += verts.size() / 3;
		vertexCount += verts.size();

		// Allocate data
		const auto triangleOffset = object->triangles.size();

		object->materialIndices.resize(object->materialIndices.size() + (verts.size() / 3), materialIdx);
		object->triangles.resize(object->triangles.size() + (verts.size() / 3));

		object->baseVertices.reserve(object->baseVertices.size() + verts.size());
		object->baseNormals.reserve(object->baseNormals.size() + verts.size());
		object->texCoords.reserve(object->texCoords.size() + verts.size());

		poses.resize(poses.size());
		for (size_t i = 0; i < poses.size(); i++)
		{
			const auto &origPose = pses.at(i);
			auto &pose = poses.at(i);

			pose.positions = origPose.positions;
			pose.normals = origPose.normals;
		}

		if (!jnts.empty())
		{
			joints.reserve(verts.size());
			weights.reserve(verts.size());

			for (int s = static_cast<int>(verts.size()), i = 0; i < s; i++)
			{
				joints.push_back(jnts.at(i));
				weights.push_back(wghts.at(i));
			}
		}

		// Add per-vertex data
		for (int s = static_cast<int>(verts.size()), i = 0; i < s; i++)
		{
			const auto idx = verts.at(i);

			object->baseVertices.emplace_back(verts.at(i), 1.0f);
			object->baseNormals.push_back(nrmls.at(i));
			if (!uvs.empty())
				object->texCoords.push_back(uvs.at(i));
			else
				object->texCoords.emplace_back(0.0f);
		}
	}

	flags &= ~INITIAL_PRIM;
}
