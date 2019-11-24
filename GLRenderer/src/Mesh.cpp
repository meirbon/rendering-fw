//
// Created by Mèir Noordermeer on 23/11/2019.
//

#include "Mesh.h"

rfw::GLMesh::GLMesh() { glGenVertexArrays(1, &VAO); }

rfw::GLMesh::~GLMesh() { glDeleteVertexArrays(1, &VAO); }

void rfw::GLMesh::setMesh(const rfw::Mesh &mesh)
{
	if (mesh.hasIndices())
	{
		hasIndices = true;
		indexBuffer.setData(mesh.indices, mesh.triangleCount * sizeof(uvec3));
	}
	CheckGL();
	vertexBuffer.setData(mesh.vertices, mesh.vertexCount * sizeof(vec4));
	CheckGL();
	vertexBuffer.setData(mesh.normals, mesh.vertexCount * sizeof(vec3));
	CheckGL();

	glBindVertexArray(VAO);
	glEnableVertexAttribArray(0);
	vertexBuffer.bind();
	glVertexAttribPointer(0, 4, GL_FLOAT, false, sizeof(vec4), nullptr);
	CheckGL();
	glEnableVertexAttribArray(1);
	normalBuffer.bind();
	glVertexAttribPointer(0, 3, GL_FLOAT, false, sizeof(vec3), nullptr);
	CheckGL();
	glBindVertexArray(0);
}
