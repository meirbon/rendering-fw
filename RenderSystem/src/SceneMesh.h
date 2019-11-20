#pragma once

#include <MathIncludes.h>

#include "utils/ArrayProxy.h"

namespace rfw
{
class SceneObject;
class MeshSkin;

struct SceneMesh
{
	struct Pose
	{
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec3> tangents;
	};

	void setPose(const rfw::MeshSkin &skin);
	void setPose(rfw::utils::ArrayProxy<float> weights);
	void setTransform(const glm::mat4 &transform);

	unsigned int matID = 0;

	unsigned int vertexOffset = 0;
	unsigned int vertexCount = 0;

	SceneObject *object = nullptr;

	std::vector<Pose> poses;
	std::vector<glm::uvec4> joints;
	std::vector<glm::vec4> weights;
};
} // namespace rfw