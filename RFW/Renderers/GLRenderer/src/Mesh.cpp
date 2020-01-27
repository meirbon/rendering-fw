#include "PCH.h"

using namespace rfw::utils;

rfw::GLMesh::GLMesh() { CheckGL(); }

rfw::GLMesh::~GLMesh()
{
	for (auto &mesh : meshes)
		delete mesh.indexBuffer;

	meshes.clear();
}

void rfw::GLMesh::setMesh(const rfw::Mesh &mesh)
{
	if (vertexBuffer.getCount() != mesh.vertexCount) // This is just an update, skip division of meshes
	{
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
				m.last = uint(mesh.triangleCount * 3 - 1);
				m.count = uint(mesh.triangleCount * 3);
				m.indexBuffer = new utils::Buffer<uint, GL_ELEMENT_ARRAY_BUFFER, GL_STATIC_DRAW>();
				m.indexBuffer->setData(mesh.indices, mesh.triangleCount * sizeof(uvec3));
				meshes.emplace_back(m);
			}
			else
			{
				SubMesh m = {};
				m.matID = lastMaterial;
				m.first = first;					 // first index
				m.last = uint(mesh.triangleCount * 3 - 1); // last index
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
				m.first = first;				  // first index
				m.last = i * 3 - 1;				  // last index
				m.count = (m.last + 1) - m.first; // vertex count
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
				m.last = uint(mesh.vertexCount - 1);
				m.count = uint(mesh.vertexCount * 3);
				meshes.emplace_back(m);
			}
			else
			{
				SubMesh m = {};
				m.matID = lastMaterial;
				m.first = first;
				m.last = uint(mesh.vertexCount - 1);
				m.count = (m.last + 1) - m.first;
				meshes.emplace_back(m);
			}
		}
		texCoordBuffer.setData(mesh.texCoords, mesh.vertexCount * sizeof(vec2));
	}

	vertexBuffer.setData(mesh.vertices, mesh.vertexCount * sizeof(vec4));
	normalBuffer.setData(mesh.normals, mesh.vertexCount * sizeof(vec3));

	vao.setBuffer(0, vertexBuffer, 4, GL_FLOAT, false, 0);
	vao.setBuffer(1, normalBuffer, 3, GL_FLOAT, false, 0);
	vao.setBuffer(2, texCoordBuffer, 2, GL_FLOAT, false, 0);
}

void rfw::GLMesh::draw(utils::GLShader &shader, uint count, const DeviceMaterial *materials, const utils::GLTexture *textures) const
{
	shader.setUniform("t0", 0);
	shader.setUniform("t1", 1);
	shader.setUniform("t2", 2);
	shader.setUniform("n0", 3);
	shader.setUniform("n1", 4);
	shader.setUniform("n2", 5);
	shader.setUniform("s", 6);
	shader.setUniform("r", 7);

	vao.bind();
	if (hasIndices)
	{
		for (const auto &mesh : meshes)
		{
			mesh.indexBuffer->bind();
			const auto &mat = materials[mesh.matID];
			const auto hostMat = reinterpret_cast<const Material *>(&mat);

			const auto flags = mat.baseData4.w;
			vec4 color_flags = vec4(hostMat->getColor(), 0);
			memcpy(&color_flags.w, &mat.baseData4.w, sizeof(unsigned int));

			shader.setUniform("color_flags", color_flags);
			shader.setUniform("parameters", mat.parameters);

			if (Material::hasFlag(flags, HasDiffuseMap))
				textures[mat.t0data4.w].bind(0);
			if (Material::hasFlag(flags, Has2ndDiffuseMap))
				textures[mat.t1data4.w].bind(1);
			if (Material::hasFlag(flags, Has3rdDiffuseMap))
				textures[mat.t2data4.w].bind(2);
			if (Material::hasFlag(flags, HasNormalMap))
				textures[mat.n0data4.w].bind(3);
			if (Material::hasFlag(flags, Has2ndNormalMap))
				textures[mat.n1data4.w].bind(4);
			if (Material::hasFlag(flags, Has3rdNormalMap))
				textures[mat.n2data4.w].bind(5);
			if (Material::hasFlag(flags, HasSpecularityMap))
				textures[mat.sdata4.w].bind(6);
			if (Material::hasFlag(flags, HasRoughnessMap))
				textures[mat.rdata4.w].bind(7);

			glDrawElementsInstanced(GL_TRIANGLES, mesh.count, GL_UNSIGNED_INT, nullptr, count);
			CheckGL();
		}
	}
	else
	{
		for (const auto &mesh : meshes)
		{
			const auto &mat = materials[mesh.matID];
			const auto hostMat = reinterpret_cast<const Material *>(&mat);

			const auto flags = mat.baseData4.w;
			vec4 color_flags = vec4(hostMat->getColor(), 0);
			memcpy(&color_flags.w, &mat.baseData4.w, sizeof(unsigned int));

			shader.setUniform("color_flags", color_flags);
			shader.setUniform("parameters", mat.parameters);

			if (Material::hasFlag(flags, HasDiffuseMap))
				textures[mat.t0data4.w].bind(0);
			if (Material::hasFlag(flags, Has2ndDiffuseMap))
				textures[mat.t1data4.w].bind(1);
			if (Material::hasFlag(flags, Has3rdDiffuseMap))
				textures[mat.t2data4.w].bind(2);
			if (Material::hasFlag(flags, HasNormalMap))
				textures[mat.n0data4.w].bind(3);
			if (Material::hasFlag(flags, Has2ndNormalMap))
				textures[mat.n1data4.w].bind(4);
			if (Material::hasFlag(flags, Has3rdNormalMap))
				textures[mat.n2data4.w].bind(5);
			if (Material::hasFlag(flags, HasSpecularityMap))
				textures[mat.sdata4.w].bind(6);
			if (Material::hasFlag(flags, HasRoughnessMap))
				textures[mat.rdata4.w].bind(7);

			glDrawArraysInstanced(GL_TRIANGLES, mesh.first, mesh.count, count);
			CheckGL();
		}
	}
}
