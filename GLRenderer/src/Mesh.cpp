//
// Created by Mèir Noordermeer on 23/11/2019.
//

#include "Mesh.h"

rfw::GLMesh::GLMesh() = default;

rfw::GLMesh::~GLMesh() = default;

void rfw::GLMesh::setMesh(const rfw::Mesh &mesh)
{
	if (mesh.hasIndices())
	{
		hasIndices = true;
		indexBuffer.setData(mesh.indices, mesh.triangleCount * sizeof(uvec3));
	}

	vertexBuffer.setData(mesh.vertices, mesh.vertexCount * sizeof(vec4));
	normalBuffer.setData(mesh.normals, mesh.vertexCount * sizeof(vec3));
	vao.setBuffer(0, vertexBuffer, 4, GL_FLOAT, false, 0);
	vao.setBuffer(1, normalBuffer, 3, GL_FLOAT, false, 0);
}
