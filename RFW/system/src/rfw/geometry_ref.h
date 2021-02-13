#pragma once

#include <cassert>
#include <tuple>
#include <vector>

#include <rfw/math.h>

#include "geometry/triangles.h"

namespace rfw
{
class system;
// Use lightweight object as a geometry reference for now, we might want to expand in the future
class geometry_ref
{
	friend class rfw::system;

  private:
	geometry_ref(size_t index, rfw::system &system) : m_Index(index), m_System(&system) { assert(m_System); }

  public:
	geometry_ref() = default;
	explicit operator size_t() const { return static_cast<size_t>(m_Index); }
	[[nodiscard]] size_t get_index() const { return m_Index; }

	[[nodiscard]] bool is_animated() const;
	void set_animation_to(float time) const;
	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const;
	[[nodiscard]] const std::vector<simd::matrix4> &get_mesh_matrices() const;
	[[nodiscard]] const std::vector<std::vector<int>> &get_light_indices() const;

  protected:
	[[nodiscard]] geometry::SceneTriangles *get_object() const;

  private:
	size_t m_Index; // Loaded geometry index
	rfw::system *m_System;
};
} // namespace rfw