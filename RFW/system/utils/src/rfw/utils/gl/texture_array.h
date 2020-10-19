#pragma once

#include <GL/glew.h>

#include <vector>

#include <glm/glm.hpp>

namespace rfw::utils
{
class texture_array
{
  public:
	explicit texture_array(unsigned int layers, unsigned int width, unsigned int height);
	texture_array(const texture_array &other);
	~texture_array();

	void bind() const;
	void bind(unsigned int slot) const;
	static void unbind();

	void set_data(unsigned int depth, const std::vector<glm::vec4> &data, unsigned int width, unsigned int height);
	void set_data(unsigned int depth, const std::vector<unsigned int> &data, unsigned int width, unsigned int height, unsigned int layer = 0);

	void set_data(unsigned int depth, const glm::vec4 *data, unsigned int width, unsigned int height, unsigned int layer = 0);
	void set_data(unsigned int depth, const unsigned int *data, unsigned int width, unsigned int height, unsigned int layer = 0);

	[[nodiscard]] GLuint get_id() const;
	void generate_mip_maps() const;

	[[nodiscard]] unsigned int get_width() const { return m_Width; }
	[[nodiscard]] unsigned int get_height() const { return m_Height; }

	explicit operator GLuint() const { return m_ID; }

  private:
	GLuint m_ID = 0, m_Width = 0, m_Height = 0;
};
} // namespace rfw::utils