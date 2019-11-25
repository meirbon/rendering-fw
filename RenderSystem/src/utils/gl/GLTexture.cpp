#include "GLTexture.h"
#include "CheckGL.h"

namespace rfw::utils
{

GLTexture::GLTexture() : m_Type(NONE), m_Width(0), m_Height(0) { glGenTextures(1, &m_ID); }

GLTexture::GLTexture(TextureType type, uint width, uint height, bool init) : m_Type(type), m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_ID);
	bind();

	if (init)
		initialize(type, width, height);
	unbind();
}

GLTexture::~GLTexture() { cleanup(); }

GLTexture::GLTexture(const GLTexture &other) : GLTexture()
{
	this->m_ID = other.m_ID;
	this->m_Width = other.m_Width;
	this->m_Height = other.m_Height;
	this->m_Type = other.m_Type;

	const GLTexture *otherPtr = &other;
	auto *ptr = const_cast<GLTexture *>(otherPtr);
	memset(ptr, 0, sizeof(GLTexture)); // Make sure other texture object does not delete this object
}

GLTexture::GLTexture(GLTexture &other)
{
	this->m_ID = other.m_ID;
	this->m_Width = other.m_Width;
	this->m_Height = other.m_Height;
	this->m_Type = other.m_Type;

	memset(&other, 0, sizeof(GLTexture));
}

void GLTexture::initialize(TextureType type, uint width, uint height)
{
	m_Type = type;
	m_Width = width;
	m_Height = height;
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(m_Type), width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	CheckGL();
}

void GLTexture::cleanup()
{
	if (m_ID)
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

void GLTexture::setData(const std::vector<glm::vec4> &data, uint width, uint height, uint layer)
{
	m_Type = VEC4;
	glTexImage2D(GL_TEXTURE_2D, layer, (uint)m_Type, width, height, 0, GL_RGBA, GL_FLOAT, data.data());
}

void GLTexture::setData(const glm::vec4 *data, uint width, uint height, uint layer)
{
	m_Type = VEC4;
	glTexImage2D(GL_TEXTURE_2D, layer, (uint)m_Type, width, height, 0, GL_RGBA, GL_FLOAT, data);
}

void GLTexture::setData(const std::vector<uint> &data, uint width, uint height, uint layer)
{
	m_Type = UINT;
	glTexImage2D(GL_TEXTURE_2D, layer, (uint)m_Type, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
}

void GLTexture::setData(const uint *data, uint width, uint height, uint layer)
{
	m_Type = UINT;
	glTexImage2D(GL_TEXTURE_2D, layer, (uint)m_Type, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
}

GLuint GLTexture::getID() const { return m_ID; }

GLuint64 GLTexture::getHandle() const { return glGetTextureHandleARB(m_ID); }

void GLTexture::generateMipMaps() const
{
	bind();
	glGenerateMipmap(GL_TEXTURE_2D);
}

} // namespace rfw::utils
