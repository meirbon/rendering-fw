#pragma once

#include <utils/gl/GLBuffer.h>
#include <utils/gl/GLShader.h>
#include <utils/gl/VertexArray.h>
#include <utils/gl/GLTexture.h>

namespace rfw
{
class GLMesh
{
	struct SubMesh
	{
		unsigned int matID = 0;
		unsigned int first = 0;
		unsigned int last = 0;
		unsigned int count = 0;
		utils::Buffer<unsigned int, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW> *indexBuffer = nullptr;
	};

  public:
	GLMesh();
	~GLMesh();

	void setMesh(const rfw::Mesh &mesh);

	bool hasIndices = false;
	utils::Buffer<vec4, GL_ARRAY_BUFFER, GL_STATIC_DRAW> vertexBuffer;
	utils::Buffer<vec3, GL_ARRAY_BUFFER, GL_STATIC_DRAW> normalBuffer;
	utils::Buffer<vec2, GL_ARRAY_BUFFER, GL_STATIC_DRAW> texCoordBuffer;

	utils::GLVertexArray vao;

	void draw(utils::GLShader &shader, unsigned int count, const DeviceMaterial *materials,
			  const utils::GLTexture *textures) const;

  private:
	std::vector<SubMesh> meshes;
};
} // namespace rfw
