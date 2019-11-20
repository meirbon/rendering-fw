#pragma once

#include <MathIncludes.h>

#include "utils/ArrayProxy.h"

#include "SceneMesh.h"

namespace rfw
{
class SceneObject;
struct SceneNode
{
	SceneNode(SceneObject *obj, std::string name, rfw::utils::ArrayProxy<int> children);

	bool update(glm::mat4 accumulatedTransform);
	void calculateTransform();

	glm::mat4 combinedTransform = glm::mat4(1.0f); // Combined transform of parent nodes
	glm::mat4 localTransform = glm::mat4(1.0f);	// T * R * S

	const std::string name = "";

	glm::vec3 translation = glm::vec3(0.0f);
	glm::quat rotation = glm::identity<glm::quat>();
	glm::vec3 scale = glm::vec3(1.0f);
	glm::mat4 matrix = glm::mat4(1.0f);

	bool transformed = true;
	bool morphed = false;
	int meshID = -1, skinID = -1;

	std::vector<float> weights;
	std::vector<int> childIndices;

  private:
	SceneObject *object = nullptr;
	bool hasUpdatedStatic = false;
};

} // namespace rfw