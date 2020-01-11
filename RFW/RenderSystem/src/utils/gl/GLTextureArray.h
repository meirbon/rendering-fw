#pragma once

#include <GL/glew.h>

#include <vector>

#include <MathIncludes.h>

namespace rfw::utils
{
class GLTexture2DArray
{
  public:
	explicit GLTexture2DArray(uint layers, uint width, uint height);
	GLTexture2DArray(const GLTexture2DArray &other);
	~GLTexture2DArray();

	void bind() const;
	void bind(uint slot) const;
	static void unbind();

	void setData(uint depth, const std::vector<glm::vec4> &data, uint width, uint height);
	void setData(uint depth, const std::vector<uint> &data, uint width, uint height, uint layer = 0);

	void setData(uint depth, const glm::vec4 *data, uint width, uint height, uint layer = 0);
	void setData(uint depth, const uint *data, uint width, uint height, uint layer = 0);

	[[nodiscard]] GLuint getID() const;
	void generateMipMaps() const;

	[[nodiscard]] uint getWidth() const { return m_Width; }
	[[nodiscard]] uint getHeight() const { return m_Height; }

	operator GLuint() const { return m_ID; }

  private:
	GLuint m_ID = 0, m_Width = 0, m_Height = 0;
};
} // namespace rfw::utils