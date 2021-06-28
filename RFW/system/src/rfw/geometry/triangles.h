#pragma once

#include <rfw/context/structs.h>
#include <rfw/utils/array_proxy.h>

namespace rfw
{
class system;
class LoadException : public std::exception
{
  public:
	explicit LoadException(std::string message) : m_Message(std::move(message)) {}

	[[nodiscard]] const char *what() const noexcept override { return m_Message.c_str(); }

  private:
	std::string m_Message;
};

namespace geometry
{

class SceneTriangles
{
  public:
	friend class rfw::system;

	SceneTriangles() = default;
	virtual ~SceneTriangles() = default;

	virtual void set_time(float timeInSeconds = 0.0f){};

	virtual utils::array_proxy<std::pair<size_t, rfw::Mesh>> get_meshes() const = 0;
	virtual utils::array_proxy<simd::matrix4> get_mesh_matrices() const = 0;
	virtual utils::array_proxy<std::vector<int>> get_light_indices(const std::vector<bool> &matLightFlags,
																   bool reinitialize = false) = 0;

	virtual std::vector<bool> get_changed_meshes() = 0;
	virtual std::vector<bool> get_changed_matrices() = 0;

	virtual utils::array_proxy<Triangle> get_triangles() = 0;
	virtual utils::array_proxy<glm::vec4> get_vertices() = 0;
	virtual bool is_animated() const { return false; }

  protected:
	virtual void prepare_meshes(rfw::system &rs) = 0;
};
} // namespace geometry
} // namespace rfw