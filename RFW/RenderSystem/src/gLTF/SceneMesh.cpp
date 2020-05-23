#include "../rfw.h"

#include "../Internal.h"

#include <tiny_gltf.h>


using namespace glm;

rfw::SceneMesh::SceneMesh() { flags |= INITIAL_PRIM; }
rfw::SceneMesh::SceneMesh(const rfw::SceneObject &obj) : object(const_cast<rfw::SceneObject *>(&obj))
{
	flags |= INITIAL_PRIM;
}

void rfw::SceneMesh::setPose(const rfw::MeshSkin &skin)
{
	using namespace glm;

	const simd::vector4 normal_mask = simd::vector4(_mm_set_epi32(0, ~0, ~0, ~0));
	auto vertices = get_vertices();
	const auto baseVertices = getBaseVertices();
	auto normals = getNormals();
	const auto baseNormals = getBaseNormals();
	auto triangles = get_triangles();

	if (flags & HAS_INDICES)
	{
		tbb::parallel_for(0, static_cast<int>(vertexCount), [&](int vIndex) {
			const uvec4 &j4 = joints[vIndex];
			const vec4 &w4 = weights[vIndex];

			rfw::simd::matrix4 skinMatrix = skin.jointMatrices[j4.x].matrix * w4.x;
			skinMatrix = skinMatrix + (skin.jointMatrices[j4.y].matrix * w4.y);
			skinMatrix = skinMatrix + (skin.jointMatrices[j4.z].matrix * w4.z);
			skinMatrix = skinMatrix + (skin.jointMatrices[j4.w].matrix * w4.w);
			simd::vector4 result = skinMatrix * baseVertices[vIndex];
			result.write_to(value_ptr(vertices[vIndex]));

			result = baseNormals[vIndex] * skinMatrix.inversed();
			result = result / result.length();
			result.write_to(value_ptr(normals[vIndex]), normal_mask);
		});

		updateTriangles();
		dirty = true;
	}
	else
	{
		tbb::parallel_for(0, static_cast<int>(faceCount), [&](int t) {
			const auto i3 = t * 3;
			auto &tri = triangles[t];

			int vIndex = i3;
			{
				const uvec4 &j4 = joints[vIndex];
				const vec4 &w4 = weights[vIndex];

				rfw::simd::matrix4 skinMatrix = skin.jointMatrices[j4.x].matrix * w4.x;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.y].matrix * w4.y;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.z].matrix * w4.z;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.w].matrix * w4.w;
				simd::vector4 result = skinMatrix * baseVertices[vIndex];
				result.write_to(value_ptr(vertices[vIndex]));

				result = baseNormals[vIndex] * skinMatrix.inversed();
				result = result / result.length();
				result.write_to(value_ptr(normals[vIndex]), normal_mask);

				tri.vertex0 = vertices[vIndex];
				tri.vN0 = normals[vIndex];
			}

			vIndex = i3 + 1;
			{
				const uvec4 &j4 = joints[vIndex];
				const vec4 &w4 = weights[vIndex];

				simd::matrix4 skinMatrix = skin.jointMatrices[j4.x].matrix * w4.x;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.y].matrix * w4.y;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.z].matrix * w4.z;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.w].matrix * w4.w;
				simd::vector4 result = skinMatrix * baseVertices[vIndex];
				result.write_to(value_ptr(vertices[vIndex]));

				result = baseNormals[vIndex] * skinMatrix.inversed();
				result = result / result.length();
				result.write_to(value_ptr(normals[vIndex]), normal_mask);

				tri.vertex1 = vertices[vIndex];
				tri.vN1 = normals[vIndex];
			}

			vIndex = i3 + 2;
			{
				const uvec4 &j4 = joints[vIndex];
				const vec4 &w4 = weights[vIndex];

				rfw::simd::matrix4 skinMatrix = skin.jointMatrices[j4.x].matrix * w4.x;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.y].matrix * w4.y;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.z].matrix * w4.z;
				skinMatrix = skinMatrix + skin.jointMatrices[j4.w].matrix * w4.w;
				simd::vector4 result = skinMatrix * baseVertices[vIndex];
				result.write_to(value_ptr(vertices[vIndex]));

				result = baseNormals[vIndex] * skinMatrix.inversed();
				result = result / result.length();
				result.write_to(value_ptr(normals[vIndex]), normal_mask);

				tri.vertex2 = vertices[vIndex];
				tri.vN2 = normals[vIndex];
			}

			const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;
		});

		updateTriangles();
		dirty = true;
	}
}

void rfw::SceneMesh::setPose(const std::vector<float> &wghts)
{
	assert(wghts.size() == poses.size() - 1);
	const auto weightCount = wghts.size();

	tbb::parallel_for(0, static_cast<int>(vertexCount), [&](int i) {
		const auto idx = i + vertexOffset;
		object->vertices[idx] = vec4(poses[0].positions[i], 1.0f);
		object->normals[idx] = poses[0].normals[i];

		for (int j = 1; j <= weightCount; j++)
		{
			const auto &pose = poses[j];

			object->vertices[idx] += wghts[j - 1] * vec4(pose.positions[i], 0);
			object->normals[idx] += wghts[j - 1] * pose.normals[i];
		}
	});

	updateTriangles();
	dirty = true;
}

void rfw::SceneMesh::setTransform(const glm::mat4 &transform)
{
	const simd::vector4 normal_mask = _mm_set_epi32(0, ~0, ~0, ~0);

	auto vertices = get_vertices();
	const auto baseVertices = getBaseVertices();
	auto normals = getNormals();
	const auto baseNormals = getBaseNormals();

	tbb::parallel_for(0, static_cast<int>(vertexCount), [&](int i) {
		vertices[i] = (transform * baseVertices[i]).vec;
		(transform * baseNormals[i]).store(value_ptr(normals[i]), normal_mask);
	});

	updateTriangles();
	dirty = true;
}

vec4 *rfw::SceneMesh::get_vertices() { return &object->vertices[vertexOffset]; }
const vec4 *rfw::SceneMesh::get_vertices() const { return &object->vertices[vertexOffset]; }

rfw::simd::vector4 *rfw::SceneMesh::getBaseVertices() { return &object->baseVertices[vertexOffset]; }
const rfw::simd::vector4 *rfw::SceneMesh::getBaseVertices() const { return &object->baseVertices[vertexOffset]; }

rfw::Triangle *rfw::SceneMesh::get_triangles() { return &object->triangles[triangleOffset]; }

const rfw::Triangle *rfw::SceneMesh::get_triangles() const { return &object->triangles[triangleOffset]; }

glm::uvec3 *rfw::SceneMesh::getIndices()
{
	if (flags & HAS_INDICES)
		return &object->indices[faceOffset];
	return nullptr;
}

const glm::uvec3 *rfw::SceneMesh::getIndices() const
{
	if (flags & HAS_INDICES)
		return &object->indices[faceOffset];
	return nullptr;
}

glm::vec2 *rfw::SceneMesh::getTexCoords() { return &object->texCoords[vertexOffset]; }

const glm::vec2 *rfw::SceneMesh::getTexCoords() const { return &object->texCoords[vertexOffset]; }

glm::vec3 *rfw::SceneMesh::getNormals() { return &object->normals[vertexOffset]; }
const glm::vec3 *rfw::SceneMesh::getNormals() const { return &object->normals[vertexOffset]; }

rfw::simd::vector4 *rfw::SceneMesh::getBaseNormals() { return &object->baseNormals[vertexOffset]; }
const rfw::simd::vector4 *rfw::SceneMesh::getBaseNormals() const { return &object->baseNormals[vertexOffset]; }

void rfw::SceneMesh::addPrimitive(const std::vector<int> &indces, const std::vector<glm::vec3> &verts,
								  const std::vector<glm::vec3> &nrmls, const std::vector<glm::vec2> &uvs,
								  const std::vector<rfw::SceneMesh::Pose> &pses, const std::vector<glm::uvec4> &jnts,
								  const std::vector<glm::vec4> &wghts, const int materialIdx)
{
	std::vector<int> indices = indces;

	if ((flags & HAS_INDICES) && indices.empty())
	{
		indices.resize(verts.size());
		for (int i = 0, s = static_cast<int>(verts.size()); i < s; i++)
			indices[i] = i;
	}

	if (!indices.empty()) // Indexed mesh stored as non-indexed
	{
#if 1
		if (flags & INITIAL_PRIM)
		{
			flags &= ~INITIAL_PRIM;
			flags |= HAS_INDICES;
			faceOffset = static_cast<uint>(object->indices.size());
			vertexOffset = static_cast<uint>(object->baseVertices.size());
			triangleOffset = static_cast<uint>(object->triangles.size());
		}

		const uint primIndexOffset = vertexCount;

		faceCount += uint(indices.size() / 3);
		vertexCount += uint(verts.size());

		// Allocate data
		const auto triangleOffset = object->triangles.size();

		object->materialIndices.resize(object->materialIndices.size() + (indices.size() / 3), materialIdx);
		object->triangles.resize(object->triangles.size() + (indices.size() / 3));

		object->baseVertices.reserve(object->baseVertices.size() + verts.size());
		object->baseNormals.reserve(object->baseNormals.size() + nrmls.size());
		object->texCoords.reserve(object->texCoords.size() + uvs.size());

		poses.resize(pses.size());
		for (size_t i = 0; i < poses.size(); i++)
		{
			const auto &origPose = pses[i];
			auto &pose = poses[i];

			pose.positions = origPose.positions;
			pose.normals = origPose.normals;
		}

		if (!jnts.empty())
		{
			joints.reserve(joints.size() + jnts.size());
			weights.reserve(weights.size() + wghts.size());

			for (int s = static_cast<int>(wghts.size()), i = 0; i < s; i++)
			{
				joints.push_back(jnts[i]);
				weights.push_back(wghts[i]);
			}
		}

		// Add per-vertex data
		for (int s = static_cast<int>(verts.size()), i = 0; i < s; i++)
		{
			object->baseVertices.emplace_back(verts[i], 1.0f);
			if (!nrmls.empty())
				object->baseNormals.push_back(nrmls[i]);
			else
				object->baseNormals.push_back(vec3(0));
			if (!uvs.empty())
				object->texCoords.push_back(uvs[i]);
			else
				object->texCoords.emplace_back(0.0f);
		}

		object->indices.reserve(object->indices.size() + (indices.size() / 3));
		for (int i = 0, s = static_cast<int>(indices.size()); i < s; i += 3)
			object->indices.push_back(uvec3(indices[i], indices[i + 1], indices[i + 2]) + primIndexOffset);
#else
		if (flags & INITIAL_PRIM)
		{
			flags &= ~INITIAL_PRIM;
			vertexOffset = static_cast<uint>(object->baseVertices.size());
			faceOffset = static_cast<uint>(object->indices.size());
		}

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
			const auto &origPose = pses[i];
			auto &pose = poses[i];

			pose.positions.reserve(indices.size());
			pose.normals.reserve(indices.size());

			for (const int idx : indices)
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
				const auto idx = indices[i];

				joints.push_back(jnts[idx]);
				weights.push_back(wghts[idx]);
			}
		}

		// Add per-vertex data
		for (int s = static_cast<int>(indices.size()), i = 0; i < s; i++)
		{
			const auto idx = indices[i];

			object->baseVertices.emplace_back(verts[idx], 1.0f);
			object->baseNormals.push_back(nrmls[idx]);
			if (!uvs.empty())
				object->texCoords.push_back(uvs[idx]);
			else
				object->texCoords.emplace_back(0.0f);
		}
#endif
	}
	else // Non-indexed mesh
	{
		if (flags & INITIAL_PRIM)
		{
			flags &= ~INITIAL_PRIM;
			triangleOffset = static_cast<uint>(object->triangles.size());
			faceOffset = static_cast<uint>(object->triangles.size());
			vertexOffset = static_cast<uint>(object->baseVertices.size());
		}

		faceCount += uint(verts.size() / 3);
		vertexCount += uint(verts.size());

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
			const auto &origPose = pses[i];
			auto &pose = poses[i];

			pose.positions = origPose.positions;
			pose.normals = origPose.normals;
		}

		if (!jnts.empty())
		{
			joints.reserve(verts.size());
			weights.reserve(verts.size());

			for (int s = static_cast<int>(verts.size()), i = 0; i < s; i++)
			{
				joints.push_back(jnts[i]);
				weights.push_back(wghts[i]);
			}
		}

		// Add per-vertex data
		for (int s = static_cast<int>(verts.size()), i = 0; i < s; i++)
		{
			object->baseVertices.emplace_back(verts[i], 1.0f);

			const simd::vector4 normal = simd::vector4(nrmls[i], 0.0f);

			if (!jnts.empty())
				object->baseNormals.push_back(normal);
			else
				object->baseNormals.push_back(normal);
			if (!uvs.empty())
				object->texCoords.push_back(uvs[i]);
			else
				object->texCoords.emplace_back(0.0f);
		}
	}

	const auto count = object->baseVertices.size() - object->vertices.size();

	object->vertices.resize(object->baseVertices.size());
	object->normals.resize(object->baseNormals.size());

	memcpy(object->vertices.data() + vertexOffset, object->baseVertices.data() + vertexOffset,
		   count * sizeof(glm::vec4));

	const simd::vector4 normal_mask = _mm_set_epi32(0, ~0, ~0, ~0);
	for (int i = int(vertexOffset), s = int(vertexOffset) + int(count); i < s; i++)
	{
		// object->baseNormals[i].write_to(value_ptr(object->normals[i]), normal_mask);
		object->normals[i] = object->baseNormals[i].vec;
	}

	flags &= ~INITIAL_PRIM;
}

void rfw::SceneMesh::updateTriangles() const
{
	if (flags & SceneMesh::HAS_INDICES)
	{
		tbb::parallel_for(0, static_cast<int>(faceCount), [&](int i) {
			const auto index = object->indices.at(i + faceOffset) + vertexOffset;
			Triangle &tri = object->triangles.at(i + triangleOffset);

			tri.vertex0 = object->vertices[index.x];
			tri.vertex1 = object->vertices[index.y];
			tri.vertex2 = object->vertices[index.z];

			tri.vN0 = object->normals[index.x];
			tri.vN1 = object->normals[index.y];
			tri.vN2 = object->normals[index.z];

			const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.material = object->materialIndices.at(i + triangleOffset);
		});
	}
	else
	{
		tbb::parallel_for(0, static_cast<int>(faceCount), [&](int i) {
			const auto idx = i * 3;
			const uvec3 index = uvec3(idx + 0, idx + 1, idx + 2) + vertexOffset;
			Triangle &tri = object->triangles.at(i + triangleOffset);

			const vec3 &v0 = object->vertices.at(index.x);
			const vec3 &v1 = object->vertices.at(index.y);
			const vec3 &v2 = object->vertices.at(index.z);

			const vec3 &n0 = object->normals.at(index.x);
			const vec3 &n1 = object->normals.at(index.y);
			const vec3 &n2 = object->normals.at(index.z);

			const vec3 N = normalize(cross(v1 - v0, v2 - v0));

			tri.vertex0 = v0;
			tri.vertex1 = v1;
			tri.vertex2 = v2;

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.vN0 = n0;
			tri.vN1 = n1;
			tri.vN2 = n2;

			tri.material = object->materialIndices.at(i + triangleOffset);
		});
	}
}
