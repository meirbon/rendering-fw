#include "instance_ref.h"

#include "system.h"

namespace rfw
{

instance_ref::instance_ref(size_t index, geometry_ref reference, rfw::system &sys)
{
	m_Members = std::make_shared<Members>(reference);
	m_Members->index = index;
	m_Members->geomReference = reference;
	m_Members->rSystem = &sys;
	assert(m_Members->rSystem);
	m_Members->translation = glm::vec3(0.0f);
	m_Members->rotation = glm::identity<glm::quat>();
	m_Members->scaling = glm::vec3(1.0f);

	const auto &meshes = reference.get_meshes();
	m_Members->instanceIDs.resize(meshes.size());
	for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
	{
		const int instanceID = static_cast<int>(sys.request_instance_index());
		sys.m_InverseInstanceMapping[instanceID] =
			std::make_tuple(static_cast<int>(index), static_cast<int>(reference.get_index()), i);
		m_Members->instanceIDs[i] = instanceID;
	}
}

void instance_ref::set_translation(const glm::vec3 value) { m_Members->translation = value; }

void instance_ref::set_rotation(const float degrees, const glm::vec3 axis)
{
	m_Members->rotation = glm::rotate(glm::identity<glm::quat>(), radians(degrees), axis);
}
void instance_ref::set_rotation(const glm::quat &q) { m_Members->rotation = q; }

void instance_ref::set_rotation(const glm::vec3 &euler) { m_Members->rotation = glm::quat(euler); }

void instance_ref::set_scaling(const glm::vec3 value) { m_Members->scaling = value; }

void instance_ref::translate(const glm::vec3 offset) { m_Members->translation = offset; }

void instance_ref::rotate(const float degrees, const glm::vec3 axis)
{
	m_Members->rotation = glm::rotate(m_Members->rotation, radians(degrees), axis);
}

void instance_ref::scale(const glm::vec3 offset) { m_Members->scaling = offset; }

void instance_ref::update() const { m_Members->rSystem->update_instance(*this, get_matrix().matrix); }

rfw::simd::matrix4 instance_ref::get_matrix() const
{
	const simd::matrix4 T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const simd::matrix4 R = glm::mat4(m_Members->rotation);
	const simd::matrix4 S = glm::scale(glm::mat4(1.0f), m_Members->scaling);
	return T * R * S;
}

glm::mat3 instance_ref::get_normal_matrix() const
{
	const simd::matrix4 T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const simd::matrix4 R = glm::mat4(m_Members->rotation);
	return mat3((T * R).inversed().matrix);
}

instance_ref::Members::Members(const geometry_ref &ref) : geomReference(ref) {}

} // namespace rfw