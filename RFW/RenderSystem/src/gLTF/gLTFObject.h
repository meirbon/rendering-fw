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
#include "Skinning.h"

namespace rfw
{

class gLTFObject : public SceneTriangles
{
  public:
	explicit gLTFObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
	~gLTFObject() = default;

	void transformTo(float timeInSeconds) override;

	Triangle *getTriangles() override;
	glm::vec4 *getVertices() override;

	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const override;
	[[nodiscard]] const std::vector<simd::matrix4> &getMeshTransforms() const override;
	[[nodiscard]] std::vector<bool> getChangedMeshes() override;
	[[nodiscard]] std::vector<bool> getChangedMeshMatrices() override;

	bool isAnimated() const override;
	const std::vector<std::vector<int>> &getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize) override;
	const std::string file;

	SceneObject scene;

  protected:
	void prepareMeshes(RenderSystem &rs) override;

  private:
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<std::pair<size_t, Mesh>> m_Meshes;

	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_HasUpdated = false;
};

} // namespace rfw