#pragma once

#include <Structures.h>

namespace rfw
{
class RenderSystem;
class LoadException : public std::exception
{
  public:
	explicit LoadException(std::string message) : m_Message(std::move(message)) {}

	[[nodiscard]] const char *what() const noexcept override { return m_Message.c_str(); }

  private:
	std::string m_Message;
};

class SceneTriangles
{
  public:
	friend class RenderSystem;

	SceneTriangles() = default;
	virtual ~SceneTriangles() = default;

	virtual void set_time(float timeInSeconds = 0.0f){};

	virtual const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const = 0;
	virtual const std::vector<simd::matrix4> &get_mesh_matrices() const = 0;
	virtual const std::vector<std::vector<int>> &get_light_indices(const std::vector<bool> &matLightFlags, bool reinitialize = false) = 0;

	virtual std::vector<bool> get_changed_meshes() = 0;
	virtual std::vector<bool> get_changed_matrices() = 0;

	virtual Triangle *get_triangles() = 0;
	virtual glm::vec4 *get_vertices() = 0;
	virtual bool is_animated() const { return false; }

  protected:
	virtual void prepare_meshes(RenderSystem &rs) = 0;
};
} // namespace rfw