#include <utils/Timer.h>
#include "SceneMesh.h"

#include "SceneObject.h"

#include "MeshSkin.h"

#define ALLOW_INDEXED_DATA 0

using namespace glm;

void rfw::SceneMesh::setPose(const rfw::MeshSkin &skin)
{
	using namespace glm;

	auto vertices = getVertices();
	const auto baseVertices = getBaseVertices();
	auto normals = getNormals();
	const auto baseNormals = getBaseNormals();

	for (int i = 0, s = static_cast<int>(vertexCount); i < s; i++)
	{
		const uvec4 &j4 = joints.at(i);
		const vec4 &w4 = weights.at(i);

		const mat4 mx = skin.jointMatrices.at(j4.x) * w4.x;
		const mat4 my = skin.jointMatrices.at(j4.y) * w4.y;
		const mat4 mz = skin.jointMatrices.at(j4.z) * w4.z;
		const mat4 mw = skin.jointMatrices.at(j4.w) * w4.w;

		const mat4 skinMatrix = mx + my + mz + mw;

		vertices[i] = skinMatrix * baseVertices[i];
		normals[i] = skinMatrix * vec4(baseNormals[i], 0);
	}

	object->updateTriangles(faceOffset, faceOffset + faceCount);
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
		for (int i = 0; i < faceCount; i++)
		{
			const auto idx = i + faceOffset;

			const auto indices = object->indices.at(idx) + vertexOffset;
			auto &tri = object->triangles.at(idx);

			tri.vertex0 = vec3(vertices[indices.x]);
			tri.vertex1 = vec3(vertices[indices.y]);
			tri.vertex2 = vec3(vertices[indices.z]);

			const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));

			tri.vN0 = normals[indices.x];
			tri.vN1 = normals[indices.y];
			tri.vN2 = normals[indices.z];

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
		object->materialIndices.resize(object->materialIndices.size() + (indices.size() / 3));
		object->triangles.resize(object->triangles.size() + (indices.size() / 3));
		object->indices.reserve(object->indices.size() + (indices.size() / 3));

		object->baseVertices.reserve(object->baseVertices.size() + indices.size());
		object->baseNormals.reserve(object->baseNormals.size() + indices.size());
		object->texCoords.reserve(object->texCoords.size() + indices.size());

		for (int i = 0; i < indices.size(); i += 3)
			object->indices.emplace_back(indices[i], indices[i + 1], indices[i + 2]);
	}
	else
	{
		faceCount += verts.size() / 3;
		object->materialIndices.resize(object->materialIndices.size() + (verts.size() / 3));
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

	if (flags & HAS_INDICES)
	{
		// Add per-face data
		for (size_t s = indices.size() / 3, triIdx = faceOffset, i = 0; i < s; i++, triIdx++)
		{
			const auto idx = object->indices.at(i + faceOffset);
			auto &tri = object->triangles.at(triIdx);
			object->materialIndices.at(triIdx) = materialIdx;

			const auto v0 = verts.at(idx.x);
			const auto v1 = verts.at(idx.y);
			const auto v2 = verts.at(idx.z);

			const auto &n0 = nrmls.at(idx.x);
			const auto &n1 = nrmls.at(idx.y);
			const auto &n2 = nrmls.at(idx.z);

			const vec3 N = normalize(cross(v1 - v0, v2 - v0));
			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.vertex0 = v0;
			tri.vertex1 = v1;
			tri.vertex2 = v2;

			tri.vN0 = n0;
			tri.vN1 = n1;
			tri.vN2 = n2;

			if (!uvs.empty())
			{
				tri.u0 = uvs.at(idx.x).x;
				tri.u1 = uvs.at(idx.y).x;
				tri.u2 = uvs.at(idx.z).x;

				tri.v0 = uvs.at(idx.x).y;
				tri.v1 = uvs.at(idx.y).y;
				tri.v2 = uvs.at(idx.z).y;
			}

			tri.material = materialIdx;
		}
	}
	else
	{
		for (int i = 0, s = static_cast<int>(verts.size()) / 3; i < s; i++)
		{
			const auto triIdx = i + faceOffset;
			auto &tri = object->triangles.at(triIdx);
			object->materialIndices.at(triIdx) = materialIdx;

			const auto v0 = verts.at(i * 3 + 0);
			const auto v1 = verts.at(i * 3 + 1);
			const auto v2 = verts.at(i * 3 + 2);

			const auto &n0 = nrmls.at(i * 3 + 0);
			const auto &n1 = nrmls.at(i * 3 + 1);
			const auto &n2 = nrmls.at(i * 3 + 2);

			const vec3 N = normalize(cross(v1 - v0, v2 - v0));
			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.vertex0 = v0;
			tri.vertex1 = v1;
			tri.vertex2 = v2;

			tri.vN0 = n0;
			tri.vN1 = n1;
			tri.vN2 = n2;

			if (!uvs.empty())
			{
				tri.u0 = uvs.at(i * 3 + 0).x;
				tri.u1 = uvs.at(i * 3 + 1).x;
				tri.u2 = uvs.at(i * 3 + 2).x;

				tri.v0 = uvs.at(i * 3 + 0).y;
				tri.v1 = uvs.at(i * 3 + 1).y;
				tri.v2 = uvs.at(i * 3 + 2).y;
			}

			tri.material = materialIdx;
		}
	}
}
rfw::SceneMesh::SceneMesh() { flags |= INITIAL_PRIM; }
