#pragma once

#include <vector>

#include "../utils/ArrayProxy.h"

#include <MathIncludes.h>
#include <Structures.h>

#include "SceneMesh.h"
#include "SceneNode.h"
#include "Skinning.h"
#include "SceneAnimation.h"

#include "../MaterialList.h"

namespace rfw
{

class SceneObject
{
  public:
	enum Flags
	{
		ALLOW_INDICES = 1
	};

	unsigned int flags = ALLOW_INDICES;

	// Render data, stored contiguously in a single vector to allow for
	//  fast data transfer to render device.
	std::vector<glm::vec4> vertices;
	std::vector<glm::vec3> normals;
	std::vector<glm::uvec3> indices;
	std::vector<glm::vec2> texCoords;

	std::vector<rfw::Triangle> triangles;
	std::vector<uint> materialIndices;

	// Original data
	std::vector<simd::vector4> baseVertices;
	std::vector<simd::vector4> baseNormals;

	std::vector<rfw::simd::matrix4> meshTranforms;
	std::vector<rfw::SceneMesh> meshes;
	std::vector<rfw::SceneNode> nodes;
	std::vector<rfw::MeshSkin> skins;
	std::vector<rfw::MeshBone> bones;
	std::vector<rfw::SceneAnimation> animations;
	std::vector<int> rootNodes;
	std::vector<bool> changedMeshNodeTransforms;

	bool dirty = true;

	bool transformTo(float timeInSeconds = 0.0f);

	void updateTriangles(uint offset = 0, uint last = 0);

	void updateTriangles(rfw::MaterialList *matList);
};

} // namespace rfw