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
	[[nodiscard]] rfw::Mesh getMesh() const override;

	[[nodiscard]] unsigned int getMatID() const { return m_MatID; }
	[[nodiscard]] std::vector<uint> getLightIndices(const std::vector<bool> &matLightFlags) const override;

	Triangle *getTriangles() override { return m_Triangles.data(); }
	glm::vec4 *getVertices() override { return m_Vertices.data(); }
	[[nodiscard]] uint getMaterialForPrim(unsigned int primitiveIdx) const override;

  private:
	std::vector<uint> m_Indices;
	std::vector<glm::vec4> m_Vertices;
	std::vector<Triangle> m_Triangles;

	unsigned int m_MatID;
};

} // namespace rfw