#pragma once

#include <MathIncludes.h>

#include "../utils/ArrayProxy.h"

#include <Structures.h>

namespace rfw
{
class SceneObject;
class MeshSkin;

struct SceneMesh
{
	SceneMesh();
	explicit SceneMesh(const SceneObject &object);

	enum Flags
	{
		INITIAL_PRIM = 1,
		HAS_INDICES = 2
	};

	struct Pose
	{
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
	};

	glm::vec3 *getNormals();
	const glm::vec3 *getNormals() const;
	glm::vec3 *getBaseNormals();
	const glm::vec3 *getBaseNormals() const;
	glm::vec4 *getVertices();
	const glm::vec4 *getVertices() const;
	glm::vec4 *getBaseVertices();
	const glm::vec4 *getBaseVertices() const;

	rfw::Triangle *getTriangles();
	const rfw::Triangle *getTriangles() const;

	glm::uvec3 *getIndices();
	const glm::uvec3 *getIndices() const;

	glm::vec2 *getTexCoords();
	const glm::vec2 *getTexCoords() const;

	void setPose(const rfw::MeshSkin &skin);
	void setPose(const std::vector<float> &weights);
	void setTransform(const glm::mat4 &transform);
	void addPrimitive(const std::vector<int> &indices, const std::vector<glm::vec3> &vertices, const std::vector<glm::vec3> &normals,
					  const std::vector<glm::vec2> &uvs, const std::vector<rfw::SceneMesh::Pose> &poses, const std::vector<glm::uvec4> &joints,
					  const std::vector<glm::vec4> &weights, const int materialIdx);
	void updateTriangles() const;

	unsigned int vertexOffset = 0;
	unsigned int vertexCount = 0;

	unsigned int faceOffset = 0;
	unsigned int faceCount = 0;

	unsigned int triangleOffset = 0;
	unsigned int flags = 0;

	SceneObject *object = nullptr;
	bool dirty = false;

	std::vector<Pose> poses;
	std::vector<glm::uvec4> joints;
	std::vector<glm::vec4> weights;
};
} // namespace rfw