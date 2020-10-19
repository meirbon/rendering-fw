#pragma once

#include <rfw_math.h>

#include <string>
#include <string_view>

#include <rfw/context/structs.h>
#include <rfw/material_list.h>
#include "../triangles.h"

#include "object.h"
#include "animation.h"
#include "mesh.h"
#include "node.h"
#include "skinning.h"

namespace rfw::geometry::gltf
{

class Object : public SceneTriangles
{
  public:
	explicit Object(std::string_view filename, material_list *matList, uint ID,
						const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
	~Object() = default;

	void set_time(float timeInSeconds) override;

	Triangle *get_triangles() override;
	glm::vec4 *get_vertices() override;

	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const override;
	[[nodiscard]] const std::vector<simd::matrix4> &get_mesh_matrices() const override;
	[[nodiscard]] std::vector<bool> get_changed_meshes() override;
	[[nodiscard]] std::vector<bool> get_changed_matrices() override;

	bool is_animated() const override;
	const std::vector<std::vector<int>> &get_light_indices(const std::vector<bool> &matLightFlags,
														   bool reinitialize) override;
	const std::string file;

	SceneObject scene;

  protected:
	void prepare_meshes(rfw::system &rs) override;

  private:
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<std::pair<size_t, Mesh>> m_Meshes;

	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_HasUpdated = false;
};

} // namespace rfw::geometry::gltf