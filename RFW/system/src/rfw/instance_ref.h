#pragma once

#include <rfw_math.h>
#include "geometry_ref.h"

#include <vector>
#include <glm/glm.hpp>

namespace rfw
{
class system;
// Use lightweight object as an instance reference for now, we might want to expand in the future
class instance_ref
{
	friend class system;

  public:
	instance_ref() = default;
	instance_ref(size_t index, geometry_ref reference, rfw::system &system);

	explicit operator size_t() const { return m_Members->index; }

	[[nodiscard]] const geometry_ref &get_geometry_ref() const { return m_Members->geomReference; }

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
		explicit Members(const geometry_ref &ref);
		glm::vec3 translation = glm::vec3(0);
		glm::quat rotation = glm::identity<glm::quat>();
		glm::vec3 scaling = glm::vec3(1);
		size_t index{};
		std::vector<size_t> instanceIDs;
		geometry_ref geomReference;
		rfw::system *rSystem = nullptr;
	};
	std::shared_ptr<Members> m_Members;
};
} // namespace rfw