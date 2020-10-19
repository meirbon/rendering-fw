#include "texture_array.h"

#include <cstring>

#include "check.h"

using uint = unsigned int;

rfw::utils::texture_array::texture_array(uint depth, uint width, uint height)
{
	glGenTextures(1, &m_ID);
	bind();
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glTexStorage3D(GL_TEXTURE_2D_ARRAY, 5, GL_RGBA8, width, height, depth);

	unbind();
	CheckGL();
}

rfw::utils::texture_array::texture_array(const texture_array &other)
{
	memcpy(this, &other, sizeof(texture_array));
	memset(const_cast<texture_array *>(&other), 0, sizeof(texture_array));
}

rfw::utils::texture_array::~texture_array()
{
	if (m_ID)
		glDeleteTextures(1, &m_ID);
	m_ID = 0;
}

void rfw::utils::texture_array::bind() const
{
	glBindTexture(GL_TEXTURE_2D_ARRAY, m_ID);
	CheckGL();
}

void rfw::utils::texture_array::bind(uint slot) const
{
	glActiveTexture(GL_TEXTURE0 + slot);
	bind();
	CheckGL();
}

void rfw::utils::texture_array::unbind()
{
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
	CheckGL();
}

void rfw::utils::texture_array::set_data(uint depth, const std::vector<glm::vec4> &data, uint width, uint height)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_FLOAT, data.data());
	unbind();
	CheckGL();
}

void rfw::utils::texture_array::set_data(uint depth, const std::vector<uint> &data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
	unbind();
	CheckGL();
}

void rfw::utils::texture_array::set_data(uint depth, const glm::vec4 *data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_FLOAT, data);
	unbind();
	CheckGL();
}

void rfw::utils::texture_array::set_data(uint depth, const uint *data, uint width, uint height, uint layer)
{
	bind();
	glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, depth, width, height, 1, GL_RGBA, GL_UNSIGNED_BYTE, data);
	unbind();
	CheckGL();
}

GLuint rfw::utils::texture_array::get_id() const { return m_ID; }

void rfw::utils::texture_array::generate_mip_maps() const
{
	glGenerateMipmap(GL_TEXTURE_2D_ARRAY);
	CheckGL();
}
