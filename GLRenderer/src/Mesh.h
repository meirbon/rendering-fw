//
// Created by Mèir Noordermeer on 23/11/2019.
//

#ifndef RENDERINGFW_GLRENDERER_SRC_MESH_H
#define RENDERINGFW_GLRENDERER_SRC_MESH_H

#include <utils/gl/CheckGL.h>
#include <utils/gl/GLBuffer.h>
#include <utils/gl/GLShader.h>

#include <Structures.h>

namespace rfw
{
class GLMesh
{
  public:
	GLMesh();
	~GLMesh();

	void setMesh(const rfw::Mesh& mesh);

	bool hasIndices = false;
	utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW> indexBuffer;
	utils::Buffer<vec4, GL_ARRAY_BUFFER, GL_STATIC_DRAW> vertexBuffer;
	utils::Buffer<vec3, GL_ARRAY_BUFFER, GL_STATIC_DRAW> normalBuffer;
	GLuint VAO = 0;

};
} // namespace rfw

#endif // RENDERINGFW_GLRENDERER_SRC_MESH_H
