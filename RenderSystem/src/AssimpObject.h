#pragma once

#include <MathIncludes.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <tuple>
#include <map>

#include <Structures.h>
#include "MaterialList.h"
#include "SceneTriangles.h"

#include "SceneObject.h"
#include "SceneAnimation.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "MeshSkin.h"

namespace rfw
{
class AssimpObject : public SceneTriangles
{
  public:
	explicit AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix = glm::identity<glm::mat4>(),
						  bool normalize = false, int material = -1);
	~AssimpObject() = default;

	void transformTo(float timeInSeconds = 0.0f) override;

	size_t traverseNode(const aiNode *node, std::map<std::string, uint> *nodeNameMapping);

	Triangle *getTriangles() override { return scene.triangles.data(); }
	glm::vec4 *getVertices() override { return scene.vertices.data(); }

	void setCurrentAnimation(uint index);

	[[nodiscard]] rfw::Mesh getMesh() const override
	{
		rfw::Mesh mesh{};
		mesh.vertices = scene.vertices.data();
		mesh.normals = scene.normals.data();
		mesh.triangles = scene.triangles.data();
		mesh.vertexCount = scene.vertices.size();
		mesh.triangleCount = scene.triangles.size();
		mesh.indices = nullptr;
		return mesh;
	}

	bool isAnimated() const override { return !scene.animations.empty(); }
	uint getAnimationCount() const override;
	void setAnimation(uint index) override;
	uint getMaterialForPrim(uint primitiveIdx) const override;

	std::vector<uint> getLightIndices(const std::vector<bool> &matLightFlags) const override;

	SceneObject scene;

  private:
	std::map<std::string, uint> m_NodeNameMapping;
	std::vector<size_t> m_NodesWithMeshes;
	std::vector<size_t> m_AssimpMeshMapping;

	std::string m_File;
	unsigned int m_CurrentAnimation = 0;
	int m_ID = -1;
	bool m_IsAnimated = false, m_HasUpdated = false;
};
} // namespace rfw