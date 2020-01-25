#pragma once

#include <GL/glew.h>

#include <vector>

#include <glm/glm.hpp>

namespace rfw::utils
{
class GLTexture2DArray
{
  public:
	explicit GLTexture2DArray(unsigned int layers, unsigned int width, unsigned int height);
	GLTexture2DArray(const GLTexture2DArray &other);
	~GLTexture2DArray();

	void bind() const;
	void bind(unsigned int slot) const;
	static void unbind();

	void setData(unsigned int depth, const std::vector<glm::vec4> &data, unsigned int width, unsigned int height);
	void setData(unsigned int depth, const std::vector<unsigned int> &data, unsigned int width, unsigned int height, unsigned int layer = 0);

	void setData(unsigned int depth, const glm::vec4 *data, unsigned int width, unsigned int height, unsigned int layer = 0);
	void setData(unsigned int depth, const unsigned int *data, unsigned int width, unsigned int height, unsigned int layer = 0);

	[[nodiscard]] GLuint getID() const;
	void generateMipMaps() const;

	[[nodiscard]] unsigned int get_width() const { return m_Width; }
	[[nodiscard]] unsigned int get_height() const { return m_Height; }

	explicit operator GLuint() const { return m_ID; }

  private:
	GLuint m_ID = 0, m_Width = 0, m_Height = 0;
};
} // namespace rfw::utils