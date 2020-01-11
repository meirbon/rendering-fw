#pragma once

namespace rfw
{
class GLMesh
{
	struct SubMesh
	{
		uint matID = 0;
		uint first = 0;
		uint last = 0;
		uint count = 0;
		utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW> *indexBuffer = nullptr;
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

	void draw(utils::GLShader &shader, uint count, const DeviceMaterial *materials, const utils::GLTexture *textures) const;

  private:
	std::vector<SubMesh> meshes;
};
} // namespace rfw
