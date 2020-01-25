#include "rfw.h"

namespace rfw
{

bool GeometryReference::is_animated() const { return m_System->m_Models[m_Index]->is_animated(); }

void GeometryReference::set_animation_to(const float time) const { m_System->set_animation_to(*this, time); }

const std::vector<std::pair<size_t, Mesh>> &GeometryReference::get_meshes() const { return m_System->m_Models[m_Index]->get_meshes(); }

const std::vector<simd::matrix4> &GeometryReference::get_mesh_matrices() const { return m_System->m_Models[m_Index]->get_mesh_matrices(); }

const std::vector<std::vector<int>> &GeometryReference::get_light_indices() const
{
	const auto lightFlags = m_System->m_Materials->getMaterialLightFlags();
	return m_System->m_Models[m_Index]->get_light_indices(lightFlags, false);
}

SceneTriangles *GeometryReference::get_object() const { return m_System->m_Models[m_Index]; }

} // namespace rfw