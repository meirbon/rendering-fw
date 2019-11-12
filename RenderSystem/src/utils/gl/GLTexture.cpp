#include "GLTexture.h"
#include "CheckGL.h"

namespace rfw::utils
{

GLTexture::GLTexture(TextureType type, uint width, uint height, bool initialize)
	: m_Type(type), m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_ID);
	bind();
	if (initialize)
	{
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(m_Type), width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
		CheckGL();
	}
	CheckGL();
	unbind();
}
GLTexture::~GLTexture() { cleanup(); }

void GLTexture::cleanup()
{
	glDeleteTextures(1, &m_ID);
	m_ID = 0;
}

void GLTexture::bind() const { glBindTexture(GL_TEXTURE_2D, m_ID); }

void GLTexture::bind(uint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	bind();
}

void GLTexture::unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

void GLTexture::setData(const std::vector<glm::vec4> &data, uint width, uint height) const
{
	glTexImage2D(GL_TEXTURE_2D, 0, (uint)m_Type, width, height, 0, GL_RGBA, GL_FLOAT, data.data());
}

void GLTexture::setData(const glm::vec4 *data, uint width, uint height) const
{
	glTexImage2D(GL_TEXTURE_2D, 0, (uint)m_Type, width, height, 0, GL_RGBA, GL_FLOAT, data);
}

void GLTexture::setData(const std::vector<uint> &data, uint width, uint height) const
{
	glTexImage2D(GL_TEXTURE_2D, 0, (uint)m_Type, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
}

void GLTexture::setData(const uint *data, uint width, uint height) const
{
	glTexImage2D(GL_TEXTURE_2D, 0, (uint)m_Type, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}

GLuint GLTexture::getID() const { return m_ID; }

GLuint64 GLTexture::getHandle() const { return glGetTextureHandleARB(m_ID); }

void GLTexture::generateMipMaps() const
{
	bind();
	glGenerateMipmap(GL_TEXTURE_2D);
}

} // namespace rfw::utils
