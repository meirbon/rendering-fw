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

	void set_time(float timeInSeconds) override;

	Triangle *get_triangles() override;
	glm::vec4 *get_vertices() override;

	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const override;
	[[nodiscard]] const std::vector<simd::matrix4> &get_mesh_matrices() const override;
	[[nodiscard]] std::vector<bool> get_changed_meshes() override;
	[[nodiscard]] std::vector<bool> get_changed_matrices() override;

	bool is_animated() const override;
	const std::vector<std::vector<int>> &get_light_indices(const std::vector<bool> &matLightFlags, bool reinitialize) override;
	const std::string file;

	SceneObject scene;

  protected:
	void prepare_meshes(RenderSystem &rs) override;

  private:
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<std::pair<size_t, Mesh>> m_Meshes;

	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_HasUpdated = false;
};

} // namespace rfw