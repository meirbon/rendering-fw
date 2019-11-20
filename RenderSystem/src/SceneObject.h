#pragma once

#include <vector>

#include "utils/ArrayProxy.h"

#include <MathIncludes.h>
#include <Structures.h>

#include "SceneMesh.h"
#include "SceneNode.h"
#include "MeshSkin.h"
#include "SceneAnimation.h"

#include "MaterialList.h"

namespace rfw
{

class SceneObject
{
  public:
	// Render data, stored contiguously in a single vector to allow for
	//  fast data transfer to render device.
	std::vector<glm::vec4> vertices;
	std::vector<glm::vec3> normals;

	std::vector<rfw::Triangle> triangles;
	std::vector<uint> materialIndices;

	// Original data
	std::vector<glm::vec4> baseVertices;
	std::vector<glm::vec3> baseNormals;

	std::vector<rfw::SceneMesh> meshes;
	std::vector<rfw::SceneNode> nodes;
	std::vector<rfw::MeshSkin> skins;
	std::vector<rfw::SceneAnimation> animations;
	std::vector<int> rootNodes;

	bool dirty = true;

	bool transformTo(float timeInSeconds = 0.0f);

	void updateTriangles(uint offset = 0, uint last = 0);

	void updateTriangles(rfw::MaterialList *matList, rfw::utils::ArrayProxy<glm::vec2> uvs);
};

} // namespace rfw