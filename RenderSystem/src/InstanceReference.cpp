#include "InstanceReference.h"

#include "RenderSystem.h"

namespace rfw
{

InstanceReference::InstanceReference(size_t index, GeometryReference reference, rfw::RenderSystem &system)
{
	m_Members = std::make_shared<Members>(reference);
	m_Members->index = index;
	m_Members->geomReference = reference;
	m_Members->rSystem = &system;
	assert(m_Members->rSystem);
	m_Members->translation = glm::vec3(0.0f);
	m_Members->rotation = glm::identity<glm::quat>();
	m_Members->scaling = glm::vec3(1.0f);

	const auto &meshes = reference.getMeshes();
	m_Members->instanceIDs.resize(meshes.size());
	for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
	{
		const int instanceID = static_cast<int>(system.requestInstanceIndex());
		system.m_InverseInstanceMapping[instanceID] = std::make_tuple(static_cast<int>(index), static_cast<int>(reference.getIndex()), i);
		m_Members->instanceIDs[i] = instanceID;
	}
}

void InstanceReference::setTranslation(const glm::vec3 value) { m_Members->translation = value; }

void InstanceReference::setRotation(const float degrees, const glm::vec3 axis)
{
	m_Members->rotation = glm::rotate(glm::identity<glm::quat>(), radians(degrees), axis);
}
void InstanceReference::setRotation(const glm::quat &q) { m_Members->rotation = q; }

void InstanceReference::setRotation(const glm::vec3 &euler) { m_Members->rotation = glm::quat(euler); }

void InstanceReference::setScaling(const glm::vec3 value) { m_Members->scaling = value; }

void InstanceReference::translate(const glm::vec3 offset) { m_Members->translation = offset; }

void InstanceReference::rotate(const float degrees, const glm::vec3 axis) { m_Members->rotation = glm::rotate(m_Members->rotation, radians(degrees), axis); }

void InstanceReference::scale(const glm::vec3 offset) { m_Members->scaling = offset; }

void InstanceReference::update() const { m_Members->rSystem->updateInstance(*this, getMatrix().matrix); }

rfw::simd::matrix4 InstanceReference::getMatrix() const
{
	const simd::matrix4 T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const simd::matrix4 R = glm::mat4(m_Members->rotation);
	const simd::matrix4 S = glm::scale(glm::mat4(1.0f), m_Members->scaling);
	return T * R * S;
}

glm::mat3 InstanceReference::getInverseMatrix() const
{
	const simd::matrix4 T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const simd::matrix4 R = glm::mat4(m_Members->rotation);
	return mat3((T * R).inversed().matrix);
}

InstanceReference::Members::Members(const GeometryReference &ref) : geomReference(ref) {}

} // namespace rfw