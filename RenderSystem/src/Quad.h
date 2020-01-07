#pragma once

#include "SceneTriangles.h"

#include "MathIncludes.h"

#include <vector>

namespace rfw
{
class Quad : public SceneTriangles
{
  public:
	Quad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, uint material);

	void transformTo(float timeInSeconds) override;

	[[nodiscard]] unsigned int getMatID() const { return m_MatID; }
	[[nodiscard]] const std::vector<std::vector<int>> &getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize) override;

	Triangle *getTriangles() override { return m_Triangles.data(); }
	glm::vec4 *getVertices() override { return m_Vertices.data(); }
	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const override;
	[[nodiscard]] const std::vector<SIMDMat4> &getMeshTransforms() const override;
	[[nodiscard]] std::vector<bool> getChangedMeshes() override { return std::vector<bool>(m_Meshes.size(), false); }
	[[nodiscard]] std::vector<bool> getChangedMeshMatrices() override { return std::vector<bool>(m_Meshes.size(), false); }

  protected:
	void prepareMeshes(RenderSystem &rs) override;

  private:
	std::vector<std::pair<size_t, rfw::Mesh>> m_Meshes;
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<SIMDMat4> m_MeshTransforms;

	std::vector<glm::vec4> m_Vertices;
	std::vector<glm::vec3> m_Normals;
	std::vector<Triangle> m_Triangles;

	unsigned int m_MatID;
};

} // namespace rfw