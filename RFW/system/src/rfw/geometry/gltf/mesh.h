#pragma once

#include <rfw/math.h>

#include <rfw/utils/array_proxy.h>

#include <rfw/context/structs.h>

namespace rfw::geometry::gltf
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

	[[nodiscard]] glm::vec3 *getNormals();
	[[nodiscard]] const glm::vec3 *getNormals() const;
	[[nodiscard]] rfw::simd::vector4 *getBaseNormals();
	[[nodiscard]] const simd::vector4 *getBaseNormals() const;
	[[nodiscard]] glm::vec4 *get_vertices();
	[[nodiscard]] const glm::vec4 *get_vertices() const;
	[[nodiscard]] rfw::simd::vector4 *getBaseVertices();
	[[nodiscard]] const simd::vector4 *getBaseVertices() const;

	[[nodiscard]] rfw::Triangle *get_triangles();
	[[nodiscard]] const rfw::Triangle *get_triangles() const;

	[[nodiscard]] glm::uvec3 *getIndices();
	[[nodiscard]] const glm::uvec3 *getIndices() const;

	[[nodiscard]] glm::vec2 *getTexCoords();
	[[nodiscard]] const glm::vec2 *getTexCoords() const;

	void set_pose(const MeshSkin &skin);
	void set_pose(const std::vector<float> &weights);
	void set_transform(const glm::mat4 &transform);
	void add_primitive(const std::vector<int> &indices, const std::vector<glm::vec3> &vertices, const std::vector<glm::vec3> &normals,
					  const std::vector<glm::vec2> &uvs, const std::vector<SceneMesh::Pose> &poses, const std::vector<glm::uvec4> &joints,
					  const std::vector<glm::vec4> &weights, int materialIdx);
	void update_triangles() const;

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