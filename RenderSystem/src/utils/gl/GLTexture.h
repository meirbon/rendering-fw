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
		UINT = GL_RGBA8UI,
		UINT4 = GL_RGBA32UI,
		INT = GL_RGBA8I,
		INT4 = GL_RGBA32I,
		VEC4 = GL_RGBA32F
	};

  public:
	GLTexture(TextureType type, uint width, uint height, bool initialize = true);
	~GLTexture();

	void cleanup();

	void bind() const;
	void bind(uint slot) const;
	static void unbind();

	void setData(const std::vector<glm::vec4> &data, uint width, uint height) const;
	void setData(const glm::vec4 *data, uint width, uint height) const;
	void setData(const std::vector<uint> &data, uint width, uint height) const;
	void setData(const uint *data, uint width, uint height) const;
	[[nodiscard]] GLuint getID() const;
	[[nodiscard]] GLuint64 getHandle() const;
	void generateMipMaps() const;

	[[nodiscard]] TextureType getType() const { return m_Type; }
	[[nodiscard]] uint getWidth() const { return m_Width; }
	[[nodiscard]] uint getHeight() const { return m_Height; }

	operator GLuint() const { return m_ID; }

  private:
	TextureType m_Type;
	GLuint m_ID;
	GLuint m_Width, m_Height;
};
} // namespace utils
} // namespace rfw
