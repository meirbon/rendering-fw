#pragma once

#include <MathIncludes.h>

#include "../utils/ArrayProxy.h"

#include "SceneMesh.h"

namespace rfw
{
struct TmpPrim
{
	std::vector<int> indices;
	std::vector<glm::vec3> normals, vertices;
	std::vector<glm::vec2> uvs;
	std::vector<glm::uvec4> joints;
	std::vector<glm::vec4> weights;
	std::vector<rfw::SceneMesh::Pose> poses;
	int matID;
};

class SceneObject;
struct SceneNode
{
	struct Transform
	{
		glm::vec3 translation = vec3(0);
		glm::quat rotation = glm::identity<glm::quat>();
		glm::vec3 scale = vec3(1);
	};

	SceneNode(SceneObject *obj, std::string name, rfw::utils::ArrayProxy<int> children, rfw::utils::ArrayProxy<int> meshIDs,
			  rfw::utils::ArrayProxy<int> skinIDs, rfw::utils::ArrayProxy<std::vector<TmpPrim>> meshes, Transform T, glm::mat4 transform);

	bool update(SIMDMat4 accumulatedTransform);
	void calculateTransform();

	SIMDMat4 combinedTransform = glm::mat4(1.0f); // Combined transform of parent nodes
	SIMDMat4 localTransform = glm::mat4(1.0f);	   // T * R * S

	const std::string name = "";

	glm::vec3 translation = glm::vec3(0.0f);
	glm::quat rotation = glm::identity<glm::quat>();
	glm::vec3 scale = glm::vec3(1.0f);
	glm::mat4 matrix = glm::mat4(1.0f);

	bool transformed = true;
	bool morphed = false;

	std::vector<int> meshIDs;
	std::vector<int> skinIDs;
	std::vector<float> weights;
	std::vector<int> childIndices;

  private:
	SceneObject *object = nullptr;
	bool hasUpdatedStatic = false;
};

} // namespace rfw