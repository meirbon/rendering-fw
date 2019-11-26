#include "GLTextureArray.h"

#include <cstring>

#include <Settings.h>
#include "CheckGL.h"

rfw::utils::GLTexture2DArray::GLTexture2DArray(uint depth, uint width, uint height)
{
	glGenTextures(1, &m_ID);
	bind();
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexStorage3D(GL_TEXTURE_2D_ARRAY, MIPLEVELCOUNT, GL_RGBA8, width, height, depth);

	unbind();
	CheckGL();
}

rfw::utils::GLTexture2DArray::GLTexture2DArray(const GLTexture2DArray &other)
{
	memcpy(this, &other, sizeof(GLTexture2DArray));
	memset(const_cast<GLTexture2DArray *>(&other), 0, sizeof(GLTexture2DArray));
}

rfw::utils::GLTexture2DArray::~GLTexture2DArray()
{
	if (m_ID)
		glDeleteTextures(1, &m_ID);
	m_ID = 0;
}

void rfw::utils::GLTexture2DArray::bind() const
{
	glBindTexture(GL_TEXTURE_2D_ARRAY, m_ID);
	CheckGL();
}

void rfw::utils::GLTexture2DArray::bind(uint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	bind();
	CheckGL();
}

void rfw::utils::GLTexture2DArray::unbind()
{
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
	CheckGL();
}

void rfw::utils::GLTexture2DArray::setData(uint depth, const std::vector<glm::vec4> &data, uint width, uint height)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_FLOAT, data.data());
	unbind();
	CheckGL();
}

void rfw::utils::GLTexture2DArray::setData(uint depth, const std::vector<uint> &data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
	unbind();
	CheckGL();
}

void rfw::utils::GLTexture2DArray::setData(uint depth, const glm::vec4 *data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_FLOAT, data);
	unbind();
	CheckGL();
}

void rfw::utils::GLTexture2DArray::setData(uint depth, const uint *data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
	unbind();
	CheckGL();
}

GLuint rfw::utils::GLTexture2DArray::getID() const { return m_ID; }

void rfw::utils::GLTexture2DArray::generateMipMaps() const
{
	glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
	CheckGL();
}
