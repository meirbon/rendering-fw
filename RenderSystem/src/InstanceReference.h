#pragma once

#include "MathIncludes.h"
#include "GeometryReference.h"

namespace rfw
{
// Use lightweight object as an instance reference for now, we might want to expand in the future
class InstanceReference
{
	friend class rfw::RenderSystem;

  public:
	InstanceReference() = default;
	InstanceReference(size_t index, GeometryReference reference, rfw::RenderSystem &system);

	explicit operator size_t() const { return m_Members->index; }

	[[nodiscard]] const GeometryReference &getGeometryReference() const { return m_Members->geomReference; }

	void setTranslation(glm::vec3 value);
	void setRotation(float degrees, glm::vec3 axis);
	void setRotation(const glm::quat &q);
	void setRotation(const glm::vec3 &euler);
	void setScaling(glm::vec3 value);

	void translate(glm::vec3 offset);
	void rotate(float degrees, glm::vec3 axis);
	void scale(glm::vec3 offset);
	void update() const;

	[[nodiscard]] size_t getIndex() const { return m_Members->index; }
	[[nodiscard]] const std::vector<size_t> &getIndices() const { return m_Members->instanceIDs; }

	[[nodiscard]] glm::mat4 getMatrix() const;
	[[nodiscard]] glm::mat3 getInverseMatrix() const;

	[[nodiscard]] glm::vec3 getScaling() const { return m_Members->scaling; }
	[[nodiscard]] glm::vec3 getRotation() const { return glm::eulerAngles(m_Members->rotation); }
	[[nodiscard]] glm::vec3 getTranslation() const { return m_Members->translation; }

  private:
	struct Members
	{
		explicit Members(const GeometryReference &ref);
		glm::vec3 translation = glm::vec3(0);
		glm::quat rotation = glm::identity<glm::quat>();
		glm::vec3 scaling = glm::vec3(1);
		size_t index{};
		std::vector<size_t> instanceIDs;
		GeometryReference geomReference;
		rfw::RenderSystem *rSystem = nullptr;
	};
	std::shared_ptr<Members> m_Members;
};
} // namespace rfw