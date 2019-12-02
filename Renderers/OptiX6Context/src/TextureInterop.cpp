#include "TextureInterop.h"

#include <cassert>

#include <utils/Logger.h>
#include "CheckCUDA.h"
#include <utils/gl/CheckGL.h>

TextureInterop::~TextureInterop()
{
	if (m_Bound)
	{
		cudaGraphicsUnmapResources(1, &m_Resource);
		m_Bound = false;
	}

	if (m_Linked)
	{
		CheckCUDA(cudaGraphicsUnregisterResource(m_Resource));
		m_Linked = false;
	}
}

void TextureInterop::setTexture(GLuint texID)
{
	if (m_Bound)
	{
		CheckCUDA(cudaGraphicsUnmapResources(1, &m_Resource));
		m_Bound = false;
	}

	if (m_Linked)
	{
		CheckCUDA(cudaGraphicsUnregisterResource(m_Resource));
		m_Linked = false;
	}

	m_TextureID = texID;
}

void TextureInterop::linkToSurface(const surfaceReference *s)
{
	m_SurfaceReference = s;
	assert(!m_Bound);
	if (m_Linked)
	{
		CheckCUDA(cudaGraphicsUnregisterResource(m_Resource));
		m_Linked = false;
	}

	glBindTexture(GL_TEXTURE_2D, m_TextureID);
	CheckGL();
	CheckCUDA(cudaGraphicsGLRegisterImage(&m_Resource, m_TextureID, GL_TEXTURE_2D,
										  cudaGraphicsRegisterFlagsSurfaceLoadStore));
	CheckGL();
	glBindTexture(GL_TEXTURE_2D, 0);
	m_Linked = true;
	CheckGL();
}

void TextureInterop::bindSurface()
{
	assert(!m_Bound);
	cudaArray *array;
	CheckCUDA(cudaGraphicsMapResources(1, &m_Resource));
	CheckCUDA(cudaGraphicsSubResourceGetMappedArray(&array, m_Resource, 0, 0));
	cudaChannelFormatDesc descriptor{};
	CheckCUDA(cudaGetChannelDesc(&descriptor, array));
	CheckCUDA(cudaBindSurfaceToArray(m_SurfaceReference, array, &descriptor));

	assert(descriptor.f != cudaChannelFormatKindNone);
	m_Bound = true;
}

void TextureInterop::unbindSurface()
{
	assert(m_Bound);
	CheckCUDA(cudaGraphicsUnmapResources(1, &m_Resource));
	m_Bound = false;
}