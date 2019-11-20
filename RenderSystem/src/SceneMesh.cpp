#include "SceneMesh.h"

#include "SceneObject.h"

#include "MeshSkin.h"

using namespace glm;

void rfw::SceneMesh::setPose(const rfw::MeshSkin &skin)
{
	using namespace glm;

	for (uint i = 0; i < vertexCount; i++)
	{
		const auto idx = i + vertexOffset;

		const uvec4 &j4 = joints.at(i);
		const vec4 &w4 = weights.at(i);

		mat4 skinMatrix = w4.x * skin.jointMatrices.at(j4.x);
		skinMatrix += w4.y * skin.jointMatrices.at(j4.y);
		skinMatrix += w4.z * skin.jointMatrices.at(j4.z);
		skinMatrix += w4.w * skin.jointMatrices.at(j4.w);

		object->vertices.at(idx) = skinMatrix * vec4(vec3(object->baseVertices.at(idx)), 1.0f);
		object->normals.at(idx) = mat3(skinMatrix) * object->baseNormals.at(idx);
	}

	object->updateTriangles(vertexOffset / 3, vertexCount / 3);
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
