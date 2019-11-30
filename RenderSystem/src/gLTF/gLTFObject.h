#pragma once

#include <MathIncludes.h>

#include <string>
#include <string_view>

#include <Structures.h>
#include "../MaterialList.h"
#include "../SceneTriangles.h"

#include "SceneObject.h"
#include "SceneAnimation.h"
#include "SceneMesh.h"
#include "SceneNode.h"
#include "MeshSkin.h"

namespace rfw
{

class gLTFObject : public SceneTriangles
{
  public:
	explicit gLTFObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
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

  private:
	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_IsAnimated = false, m_HasUpdated = false;
};

} // namespace rfw