#define GLM_FORCE_SIMD_AVX2
#include "SceneObject.h"

using namespace rfw;

bool rfw::SceneObject::transformTo(float timeInSeconds)
{
	vertices.resize(baseVertices.size());
	normals.resize(baseNormals.size());

	for (auto &anim : animations)
		anim.setTime(timeInSeconds);

	bool changed = false;

	for (auto idx : rootNodes)
	{
		auto matrix = glm::identity<glm::mat4>();
		changed |= nodes.at(idx).update(matrix);
	}

	return changed;
}

void rfw::SceneObject::updateTriangles(uint offset, uint last)
{
	if (last == 0)
		last = triangles.size();

	assert(last <= triangles.size());

	if (indices.empty())
	{
		for (int i = static_cast<int>(offset), s = static_cast<int>(last); i < s; i++)
		{
			const auto idx = i * 3;
			Triangle &tri = triangles.at(i);
			const vec3 &v0 = vertices.at(idx + 0);
			const vec3 &v1 = vertices.at(idx + 1);
			const vec3 &v2 = vertices.at(idx + 2);

			tri.vertex0 = v0;
			tri.vertex1 = v1;
			tri.vertex2 = v2;

			const vec3 &n0 = normals.at(idx + 0);
			const vec3 &n1 = normals.at(idx + 1);
			const vec3 &n2 = normals.at(idx + 2);

			vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			if (dot(N, n0) < 0.0f && dot(N, n1) < 0.0f && dot(N, n1) < 0.0f)
				N *= -1.0f; // flip if not consistent with vertex normals

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.vN0 = n0;
			tri.vN1 = n1;
			tri.vN2 = n2;

			tri.material = materialIndices.at(i);
		}
	}
	else
	{
		for (int i = static_cast<int>(offset), s = static_cast<int>(last); i < s; i++)
		{
			const auto index = indices.at(i);
			Triangle &tri = triangles.at(i);
			const vec3 &v0 = vertices.at(index.x);
			const vec3 &v1 = vertices.at(index.y);
			const vec3 &v2 = vertices.at(index.z);

			const vec3 &n0 = normals.at(index.x);
			const vec3 &n1 = normals.at(index.y);
			const vec3 &n2 = normals.at(index.z);

			vec3 N = normalize(cross(v1 - v0, v2 - v0));

			if (dot(N, n0) < 0.0f && dot(N, n1) < 0.0f && dot(N, n1) < 0.0f)
				N *= -1.0f; // flip if not consistent with vertex normals

			tri.vertex0 = v0;
			tri.vertex1 = v1;
			tri.vertex2 = v2;

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.vN0 = n0;
			tri.vN1 = n1;
			tri.vN2 = n2;

			tri.material = materialIndices.at(i);
		}
	}
}

void rfw::SceneObject::updateTriangles(rfw::MaterialList *matList)
{
	if (indices.empty())
	{
		for (uint i = 0, s = static_cast<uint>(triangles.size()); i < s; i++)
		{
			const auto idx = i * 3;
			Triangle &tri = triangles.at(i);

			if (!texCoords.empty())
			{
				tri.u0 = texCoords.at(idx + 0).x;
				tri.v0 = texCoords.at(idx + 0).y;

				tri.u1 = texCoords.at(idx + 1).x;
				tri.v1 = texCoords.at(idx + 1).y;

				tri.u2 = texCoords.at(idx + 2).x;
				tri.v2 = texCoords.at(idx + 2).y;
			}

			const HostMaterial &mat = matList->get(tri.material);
			int texID = mat.map[0].textureID;
			if (texID > -1)
			{
				const Texture &texture = matList->getTextures().at(texID);

				const float Ta =
					static_cast<float>(texture.width * texture.height) * abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
				const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
				tri.LOD = 0.5f * log2f(Ta / Pa);
			}
		}
	}
	else
	{
		for (uint i = 0, s = static_cast<uint>(triangles.size()); i < s; i++)
		{
			const auto index = indices.at(i);
			Triangle &tri = triangles.at(i);

			if (!texCoords.empty())
			{
				tri.u0 = texCoords.at(index.x).x;
				tri.v0 = texCoords.at(index.x).y;

				tri.u1 = texCoords.at(index.y).x;
				tri.v1 = texCoords.at(index.y).y;

				tri.u2 = texCoords.at(index.z).x;
				tri.v2 = texCoords.at(index.z).y;
			}

			const HostMaterial &mat = matList->get(tri.material);
			int texID = mat.map[0].textureID;
			if (texID > -1)
			{
				const Texture &texture = matList->getTextures().at(texID);

				const float Ta =
					static_cast<float>(texture.width * texture.height) * abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
				const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
				tri.LOD = 0.5f * log2f(Ta / Pa);
			}
		}
	}
}
