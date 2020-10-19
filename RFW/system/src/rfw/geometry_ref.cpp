#include "rfw.h"

namespace rfw
{

bool geometry_ref::is_animated() const { return m_System->m_Models[m_Index]->is_animated(); }

void geometry_ref::set_animation_to(const float time) const { m_System->set_animation_to(*this, time); }

const std::vector<std::pair<size_t, Mesh>> &geometry_ref::get_meshes() const { return m_System->m_Models[m_Index]->get_meshes(); }

const std::vector<simd::matrix4> &geometry_ref::get_mesh_matrices() const { return m_System->m_Models[m_Index]->get_mesh_matrices(); }

const std::vector<std::vector<int>> &geometry_ref::get_light_indices() const
{
	const auto lightFlags = m_System->m_Materials->get_material_light_flags();
	return m_System->m_Models[m_Index]->get_light_indices(lightFlags, false);
}

geometry::SceneTriangles *geometry_ref::get_object() const { return m_System->m_Models[m_Index]; }

} // namespace rfw