#pragma once

#include "triangles.h"

#include <rfw/math.h>

#include <vector>

namespace rfw::geometry
{
class Quad : public SceneTriangles
{
  public:
	Quad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, uint material);

	void set_time(float timeInSeconds) override;

	[[nodiscard]] unsigned int getMatID() const { return m_MatID; }
	[[nodiscard]] const std::vector<std::vector<int>> &get_light_indices(const std::vector<bool> &matLightFlags, bool reinitialize) override;

	Triangle *get_triangles() override { return m_Triangles.data(); }
	glm::vec4 *get_vertices() override { return m_Vertices.data(); }
	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const override;
	[[nodiscard]] const std::vector<simd::matrix4> &get_mesh_matrices() const override;
	[[nodiscard]] std::vector<bool> get_changed_meshes() override { return std::vector<bool>(m_Meshes.size(), false); }
	[[nodiscard]] std::vector<bool> get_changed_matrices() override { return std::vector<bool>(m_Meshes.size(), false); }

  protected:
	void prepare_meshes(rfw::system &rs) override;

  private:
	std::vector<std::pair<size_t, rfw::Mesh>> m_Meshes;
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<simd::matrix4> m_MeshTransforms;

	std::vector<glm::vec4> m_Vertices;
	std::vector<glm::vec3> m_Normals;
	std::vector<Triangle> m_Triangles;

	unsigned int m_MatID;
};

} // namespace rfw