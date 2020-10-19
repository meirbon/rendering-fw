#include "texture.h"

#include "check.h"

using uint = unsigned int;

namespace rfw::utils
{

texture::texture()
{
	glGenTextures(1, &m_ID);
	CheckGL();
	bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	unbind();
	CheckGL();

	assert(m_ID > 0);
}

texture::texture(TextureType type, uint width, uint height, bool init) : m_Type(type), m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_ID);
	bind();

	if (init)
		initialize(type, width, height);
	unbind();
	CheckGL();
	assert(m_ID > 0);
}

texture::~texture() { cleanup(); }

texture::texture(const texture &other)
{
	if (m_ID > 0)
		glDeleteTextures(1, &m_ID);

	m_ID = other.m_ID;
	m_Width = other.m_Width;
	m_Height = other.m_Height;
	m_Type = other.m_Type;

	const texture *otherPtr = &other;
	auto *ptr = const_cast<texture *>(otherPtr);
	memset(ptr, 0, sizeof(texture)); // Make sure other texture object does not delete this object
}

void texture::initialize(TextureType type, uint width, uint height)
{
	m_Type = type;
	m_Width = width;
	m_Height = height;
	bind();
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	CheckGL();
	glTexImage2D(GL_TEXTURE_2D, 0, static_cast<GLint>(m_Type), width, height, 0, GL_RGBA, GL_FLOAT, nullptr);
	CheckGL();
	unbind();
}

void texture::cleanup()
{
	if (m_ID)
		glDeleteTextures(1, &m_ID);
	m_ID = 0;
}

void texture::bind() const
{
	glBindTexture(GL_TEXTURE_2D, m_ID);
	CheckGL();
}

void texture::bind(uint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	bind();
	CheckGL();
}

void texture::unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

void texture::setData(const std::vector<glm::vec4> &data, uint width, uint height, uint layer)
{
	setData(data.data(), width, height, layer);
}

void texture::setData(const glm::vec4 *data, uint width, uint height, uint layer)
{
	CheckGL();
	m_Width = width;
	m_Height = height;
	m_Type = VEC4;
	bind();
	glTexImage2D(GL_TEXTURE_2D, layer, static_cast<GLint>(m_Type), width, height, 0, GL_RGBA, GL_FLOAT, data);
	CheckGL();
	unbind();
}

void texture::setData(const std::vector<uint> &data, uint width, uint height, uint layer)
{
	setData(data.data(), width, height, layer);
}

void texture::setData(const uint *data, uint width, uint height, uint layer)
{
	CheckGL();
	m_Width = width;
	m_Height = height;
	m_Type = UINT;
	bind();
	glTexImage2D(GL_TEXTURE_2D, layer, static_cast<GLint>(m_Type), width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
	CheckGL();
	unbind();
}

GLuint texture::getID() const { return m_ID; }

GLuint64 texture::getHandle() const { return glGetTextureHandleARB(m_ID); }

void texture::generateMipMaps() const
{
	bind();
	glGenerateMipmap(GL_TEXTURE_2D);
	CheckGL();
	unbind();
}

} // namespace rfw::utils
