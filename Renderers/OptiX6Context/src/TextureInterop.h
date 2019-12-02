#pragma once

#include <GL/glew.h>
#include <cuda_gl_interop.h>

class TextureInterop
{
  public:
	~TextureInterop();

	void setTexture(GLuint texID);

	cudaGraphicsResource **getResourceID() { return &m_Resource; }
	void linkToSurface(const surfaceReference *s);
	void bindSurface();
	void unbindSurface();

	const surfaceReference *getSurfaceReference() const { return m_SurfaceReference; }
	const cudaGraphicsResource *getResource() const { return m_Resource; }

  private:
	GLuint m_TextureID = 0;
	cudaGraphicsResource *m_Resource = nullptr;
	const surfaceReference *m_SurfaceReference = nullptr;
	bool m_Linked = false;
	bool m_Bound = false;
};