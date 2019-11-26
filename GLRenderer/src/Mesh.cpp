//
// Created by Mèir Noordermeer on 23/11/2019.
//

#include "Mesh.h"

rfw::GLMesh::GLMesh() { CheckGL(); }

rfw::GLMesh::~GLMesh()
{
	for (auto &mesh : meshes)
		delete mesh.indexBuffer;

	meshes.clear();
}

void rfw::GLMesh::setMesh(const rfw::Mesh &mesh)
{
	vec2 *uvs = const_cast<vec2 *>(mesh.texCoords);

	if (mesh.hasIndices())
	{
		hasIndices = true;
		auto lastMaterial = mesh.triangles[0].material;
		int first = 0;

		for (int i = 0; i < mesh.triangleCount; i++)
		{
			// Keep iterating until we find a triangle with a different material
			const auto &tri = mesh.triangles[i];
			if (tri.material == lastMaterial)
				continue;

			// Store mesh with offsets
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = first;				  // first index
			m.last = i * 3 - 1;				  // last index
			m.count = (m.last + 1) - m.first; // vertex count
			m.indexBuffer = new utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW>();
			m.indexBuffer->setData(mesh.indices + (m.first / 3), m.count * sizeof(unsigned int));
			meshes.emplace_back(m);

			lastMaterial = tri.material;

			first = i * 3; // update first
		}

		// No different materials found, we can draw this mesh in one go
		if (meshes.empty())
		{
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = 0;
			m.last = mesh.triangleCount * 3 - 1;
			m.count = mesh.triangleCount * 3;
			m.indexBuffer = new utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW>();
			m.indexBuffer->setData(mesh.indices, mesh.triangleCount * sizeof(uvec3));
			meshes.emplace_back(m);
		}
		else
		{
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = first;					 // first index
			m.last = mesh.triangleCount * 3 - 1; // last index
			m.count = (m.last + 1) - m.first;	 // vertex count
			m.indexBuffer = new utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW>();
			m.indexBuffer->setData(mesh.indices + (m.first / 3), m.count * sizeof(unsigned int));
			meshes.emplace_back(m);
		}
	}
	else
	{
		auto lastMaterial = mesh.triangles[0].material;
		int first = 0;

		// Loop over number of triangles
		for (int i = 0; i < mesh.triangleCount; i++)
		{
			// Keep iterating until we find a triangle with a different material
			const auto &tri = mesh.triangles[i];
			if (tri.material == lastMaterial)
				continue;

			// Store mesh with offsets
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = first;						// first index
			m.last = i * 3 - 1;						// last index
			m.count = i /* m.last + 1 */ - m.first; // vertex count
			meshes.emplace_back(m);

			lastMaterial = tri.material;

			first = i * 3; // update first
		}

		// No different materials found, we can draw this mesh in one go
		if (meshes.empty())
		{
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = 0;
			m.last = mesh.vertexCount - 1;
			m.count = mesh.vertexCount * 3;
			meshes.emplace_back(m);
		}
		else
		{
			SubMesh m = {};
			m.matID = lastMaterial;
			m.first = first;
			m.last = mesh.vertexCount - 1;
			m.count = (m.last + 1) - m.first;
			meshes.emplace_back(m);
		}
	}

	vertexBuffer.setData(mesh.vertices, mesh.vertexCount * sizeof(vec4));
	normalBuffer.setData(mesh.normals, mesh.vertexCount * sizeof(vec3));
	if (mesh.hasTexCoords())
		texCoordBuffer.setData(uvs, mesh.vertexCount * sizeof(vec2));
	else
	{
		uvs = new vec2[mesh.vertexCount];
		texCoordBuffer.setData(uvs, mesh.vertexCount * sizeof(vec2));
		delete[] uvs;
	}

	vao.setBuffer(0, vertexBuffer, 4, GL_FLOAT, false, 0);
	vao.setBuffer(1, normalBuffer, 3, GL_FLOAT, false, 0);
	vao.setBuffer(2, texCoordBuffer, 2, GL_FLOAT, false, 0);
}

void rfw::GLMesh::draw(utils::GLShader &shader, uint count) const
{
	vao.bind();
	if (hasIndices)
	{
		for (const auto &mesh : meshes)
		{
			mesh.indexBuffer->bind();
			glDrawElementsInstanced(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, nullptr, count);
		}
	}
	else
	{
		for (const auto &mesh : meshes)
		{
			glDrawArraysInstanced(GL_TRIANGLES, mesh.first, mesh.count, count);
		}
	}
	CheckGL();
}
