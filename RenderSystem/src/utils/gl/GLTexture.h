#pragma once

#include <GL/glew.h>
#include <vector>
#include "MathIncludes.h"

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
	GLTexture(TextureType type, uint width, uint height, bool initialize = true);
	~GLTexture();
	GLTexture(const GLTexture &other);

	void initialize(TextureType type, uint width, uint height);
	void cleanup();

	void bind() const;
	void bind(uint slot) const;
	static void unbind();

	void setData(const std::vector<glm::vec4> &data, uint width, uint height, uint layer = 0);
	void setData(const std::vector<uint> &data, uint width, uint height, uint layer = 0);

	void setData(const glm::vec4 *data, uint width, uint height, uint layer = 0);
	void setData(const uint *data, uint width, uint height, uint layer = 0);

	[[nodiscard]] GLuint getID() const;
	[[nodiscard]] GLuint64 getHandle() const;
	void generateMipMaps() const;

	[[nodiscard]] TextureType getType() const { return m_Type; }
	[[nodiscard]] uint getWidth() const { return m_Width; }
	[[nodiscard]] uint getHeight() const { return m_Height; }

	operator GLuint() const { return m_ID; }

  private:
	TextureType m_Type = NONE;
	GLuint m_ID = 0;
	GLuint m_Width = 0, m_Height = 0;
};
} // namespace utils
} // namespace rfw
