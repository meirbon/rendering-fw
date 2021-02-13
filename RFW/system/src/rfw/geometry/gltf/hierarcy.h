#pragma once

#include <vector>

#include <rfw/utils/array_proxy.h>

#include <rfw/math.h>
#include <rfw/context/structs.h>

#include "mesh.h"
#include "node.h"
#include "skinning.h"
#include "animation.h"

#include <rfw/material_list.h>

namespace rfw::geometry::gltf
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
	std::vector<SceneMesh> meshes;
	std::vector<SceneNode> nodes;
	std::vector<MeshSkin> skins;
	std::vector<MeshBone> bones;
	std::vector<SceneAnimation> animations;
	std::vector<int> rootNodes;
	std::vector<bool> changedMeshNodeTransforms;

	bool dirty = true;

	bool set_time(float timeInSeconds = 0.0f);

	void updateTriangles(uint offset = 0, uint last = 0);

	void updateTriangles(rfw::material_list *matList);
};

} // namespace rfw