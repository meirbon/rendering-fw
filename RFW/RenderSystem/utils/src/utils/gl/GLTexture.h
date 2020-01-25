#pragma once

#include <GL/glew.h>
#include <vector>

#include <glm/glm.hpp>

namespace rfw
{
class RenderSystem;
namespace utils
{
class GLTexture
{
  public:
	// Rendersystem needs access to private members
	friend class rfw::RenderSystem;
	enum TextureType
	{
		NONE = 0,
		UINT = GL_RGBA,
		UINT4 = GL_RGBA,
		INT = GL_RGBA,
		INT4 = GL_RGBA,
		VEC4 = GL_RGBA32F
	};

  public:
	GLTexture();
	GLTexture(TextureType type, unsigned int width, unsigned int height, bool initialize = true);
	~GLTexture();
	GLTexture(const GLTexture &other);

	void initialize(TextureType type, unsigned int width, unsigned int height);
	void cleanup();

	void bind() const;
	void bind(unsigned int slot) const;
	static void unbind();

	void setData(const std::vector<glm::vec4> &data, unsigned int width, unsigned int height, unsigned int layer = 0);
	void setData(const std::vector<unsigned int> &data, unsigned int width, unsigned int height, unsigned int layer = 0);

	void setData(const glm::vec4 *data, unsigned int width, unsigned int height, unsigned int layer = 0);
	void setData(const unsigned int *data, unsigned int width, unsigned int height, unsigned int layer = 0);

	[[nodiscard]] GLuint getID() const;
	[[nodiscard]] GLuint64 getHandle() const;
	void generateMipMaps() const;

	[[nodiscard]] TextureType getType() const { return m_Type; }
	[[nodiscard]] unsigned int get_width() const { return m_Width; }
	[[nodiscard]] unsigned int get_height() const { return m_Height; }

	explicit operator GLuint() const { return m_ID; }

  private:
	TextureType m_Type = NONE;
	GLuint m_ID = 0;
	GLuint m_Width = 0, m_Height = 0;
};
} // namespace utils
} // namespace rfw
