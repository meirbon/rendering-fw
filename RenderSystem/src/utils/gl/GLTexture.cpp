#include "../../rfw.h"

namespace rfw::utils
{

GLTexture::GLTexture()
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

GLTexture::GLTexture(TextureType type, uint width, uint height, bool init) : m_Type(type), m_Width(width), m_Height(height)
{
	glGenTextures(1, &m_ID);
	bind();

	if (init)
		initialize(type, width, height);
	unbind();
	CheckGL();
	assert(m_ID > 0);
}

GLTexture::~GLTexture() { cleanup(); }

GLTexture::GLTexture(const GLTexture &other)
{
	if (m_ID > 0)
		glDeleteTextures(1, &m_ID);

	m_ID = other.m_ID;
	m_Width = other.m_Width;
	m_Height = other.m_Height;
	m_Type = other.m_Type;

	const GLTexture *otherPtr = &other;
	auto *ptr = const_cast<GLTexture *>(otherPtr);
	memset(ptr, 0, sizeof(GLTexture)); // Make sure other texture object does not delete this object
}

void GLTexture::initialize(TextureType type, uint width, uint height)
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

void GLTexture::cleanup()
{
	if (m_ID)
		glDeleteTextures(1, &m_ID);
	m_ID = 0;
}

void GLTexture::bind() const
{
	glBindTexture(GL_TEXTURE_2D, m_ID);
	CheckGL();
}

void GLTexture::bind(uint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	bind();
	CheckGL();
}

void GLTexture::unbind() { glBindTexture(GL_TEXTURE_2D, 0); }

void GLTexture::setData(const std::vector<glm::vec4> &data, uint width, uint height, uint layer) { setData(data.data(), width, height, layer); }

void GLTexture::setData(const glm::vec4 *data, uint width, uint height, uint layer)
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

void GLTexture::setData(const std::vector<uint> &data, uint width, uint height, uint layer) { setData(data.data(), width, height, layer); }

void GLTexture::setData(const uint *data, uint width, uint height, uint layer)
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

GLuint GLTexture::getID() const { return m_ID; }

GLuint64 GLTexture::getHandle() const { return glGetTextureHandleARB(m_ID); }

void GLTexture::generateMipMaps() const
{
	bind();
	glGenerateMipmap(GL_TEXTURE_2D);
	CheckGL();
	unbind();
}

} // namespace rfw::utils
