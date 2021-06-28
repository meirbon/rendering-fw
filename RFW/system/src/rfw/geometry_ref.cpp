#include "rfw.h"

namespace rfw
{

bool geometry_ref::is_animated() const { return m_System->m_Models[m_Index]->is_animated(); }

void geometry_ref::set_animation_to(const float time) const { m_System->set_animation_to(*this, time); }

utils::array_proxy<std::pair<size_t, Mesh>> geometry_ref::get_meshes() const
{
	return m_System->m_Models[m_Index]->get_meshes();
}

utils::array_proxy<simd::matrix4> geometry_ref::get_mesh_matrices() const
{
	return m_System->m_Models[m_Index]->get_mesh_matrices();
}

utils::array_proxy<std::vector<int>> geometry_ref::get_light_indices() const
{
	const auto lightFlags = m_System->m_Materials->get_material_light_flags();
	return m_System->m_Models[m_Index]->get_light_indices(lightFlags, false);
}

geometry::SceneTriangles *geometry_ref::get_object() const { return m_System->m_Models[m_Index]; }

} // namespace rfw