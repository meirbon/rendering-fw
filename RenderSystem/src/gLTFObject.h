#pragma once

#include <MathIncludes.h>

#include <string>
#include <string_view>

#include <Structures.h>
#include "MaterialList.h"
#include "SceneTriangles.h"

#include "SceneAnimation.h"

#include "SceneObject.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "MeshSkin.h"

namespace rfw
{

class gLTFObject : public SceneTriangles
{
  public:
	explicit gLTFObject(std::string_view filename, MaterialList *matList, uint ID,
						const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
	~gLTFObject() = default;

	void transformTo(float timeInSeconds = 0.0f) override;

	Triangle *getTriangles() override;
	glm::vec4 *getVertices() override;

	[[nodiscard]] rfw::Mesh getMesh() const override;

	bool isAnimated() const override;
	uint getAnimationCount() const override;
	void setAnimation(uint index) override;
	uint getMaterialForPrim(uint primitiveIdx) const override;

	std::vector<uint> getLightIndices(const std::vector<bool> &matLightFlags) const override;

	const std::string file;

	SceneObject scene;

	std::vector<glm::vec2> texCoords;

  private:
	void addPrimitive(rfw::SceneMesh &mesh, const std::vector<int> &indices, const std::vector<glm::vec3> &vertices,
					  const std::vector<glm::vec3> &normals, const std::vector<glm::vec2> &uvs,
					  const std::vector<rfw::SceneMesh::Pose> &poses, const std::vector<glm::uvec4> &joints,
					  const std::vector<glm::vec4> &weights, int materialIdx);

	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_IsAnimated = false, m_HasUpdated = false;
};

} // namespace rfw