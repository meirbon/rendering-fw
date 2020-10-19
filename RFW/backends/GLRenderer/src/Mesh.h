#pragma once

#include <rfw/utils/gl/buffer.h>
#include <rfw/utils/gl/shader.h>
#include <rfw/utils/gl/vertex_array.h>
#include <rfw/utils/gl/texture.h>

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
		utils::buffer<unsigned int, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW> *indexBuffer = nullptr;
	};

  public:
	GLMesh();
	~GLMesh();

	void setMesh(const rfw::Mesh &mesh);

	bool hasIndices = false;
	utils::buffer<vec4, GL_ARRAY_BUFFER, GL_STATIC_DRAW> vertexBuffer;
	utils::buffer<vec3, GL_ARRAY_BUFFER, GL_STATIC_DRAW> normalBuffer;
	utils::buffer<vec2, GL_ARRAY_BUFFER, GL_STATIC_DRAW> texCoordBuffer;

	utils::vertex_array vao;

	void draw(utils::shader &shader, unsigned int count, const DeviceMaterial *materials,
			  const utils::texture *textures) const;

  private:
	std::vector<SubMesh> meshes;
};
} // namespace rfw
