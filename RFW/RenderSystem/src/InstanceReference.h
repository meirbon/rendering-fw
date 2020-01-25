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

	[[nodiscard]] const GeometryReference &get_geometry_ref() const { return m_Members->geomReference; }

	void set_translation(glm::vec3 value);
	void set_rotation(float degrees, glm::vec3 axis);
	void set_rotation(const glm::quat &q);
	void set_rotation(const glm::vec3 &euler);
	void set_scaling(glm::vec3 value);

	void translate(glm::vec3 offset);
	void rotate(float degrees, glm::vec3 axis);
	void scale(glm::vec3 offset);
	void update() const;

	[[nodiscard]] size_t get_index() const { return m_Members->index; }
	[[nodiscard]] const std::vector<size_t> &getIndices() const { return m_Members->instanceIDs; }

	[[nodiscard]] simd::matrix4 get_matrix() const;
	[[nodiscard]] glm::mat3 get_normal_matrix() const;

	[[nodiscard]] glm::vec3 get_scaling() const { return m_Members->scaling; }
	[[nodiscard]] glm::vec3 get_rotation() const { return glm::eulerAngles(m_Members->rotation); }
	[[nodiscard]] glm::vec3 get_translation() const { return m_Members->translation; }

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