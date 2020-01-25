#include "../rfw.h"

#include "../Internal.h"

#include <tiny_gltf.h>

using namespace rfw;

bool rfw::SceneObject::set_time(float timeInSeconds)
{
	vertices.resize(baseVertices.size());
	normals.resize(baseNormals.size());

	for (auto &anim : animations)
		anim.setTime(timeInSeconds);

	bool changed = false;

	changedMeshNodeTransforms.resize(meshes.size());

	for (auto idx : rootNodes)
		changed |= nodes.at(idx).update(mat4(1.0f));

	return changed;
}

void rfw::SceneObject::updateTriangles(uint offset, uint last)
{
	if (last == 0)
		last = static_cast<uint>(triangles.size());

	assert(last <= triangles.size());

	if (indices.empty())
	{

#if USE_PARALLEL_FOR
		concurrency::parallel_for<int>(offset, static_cast<int>(last), [&](int i) {
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
		});
#else
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
#endif
	}
	else
	{

#if USE_PARALLEL_FOR
		concurrency::parallel_for<int>(0, static_cast<int>(meshes.size()), [&](int meshID) {
			const auto &mesh = meshes[meshID];

			if (mesh.flags & SceneMesh::HAS_INDICES)
			{
				for (int i = 0; i < mesh.faceCount; i++)
				{
					const auto index = indices.at(i + mesh.faceOffset) + mesh.vertexOffset;
					Triangle &tri = triangles.at(i + mesh.triangleOffset);

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

					tri.material = materialIndices.at(i + mesh.triangleOffset);
				}
			}
			else
			{
				for (int i = 0; i < mesh.faceCount; i++)
				{
					const auto idx = i * 3;
					const uvec3 index = uvec3(idx + 0, idx + 1, idx + 2) + mesh.vertexOffset;
					Triangle &tri = triangles.at(i + mesh.triangleOffset);

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

					tri.material = materialIndices.at(i + mesh.triangleOffset);
				}
			}
		});
#else

		for (int meshID = 0, sm = static_cast<int>(meshes.size()); meshID < sm; meshID++)
		{
			const auto &mesh = meshes[meshID];

			if (mesh.flags & SceneMesh::HAS_INDICES)
			{
				for (int i = 0; i < mesh.faceCount; i++)
				{
					const auto index = indices.at(i + mesh.faceOffset) + mesh.vertexOffset;
					Triangle &tri = triangles.at(i + mesh.triangleOffset);

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

					tri.material = materialIndices.at(i + mesh.triangleOffset);
				}
			}
			else
			{
				for (int i = 0; i < mesh.faceCount; i++)
				{
					const auto idx = i * 3;
					const uvec3 index = uvec3(idx + 0, idx + 1, idx + 2) + mesh.vertexOffset;
					Triangle &tri = triangles.at(i + mesh.triangleOffset);

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

					tri.material = materialIndices.at(i + mesh.triangleOffset);
				}
			}
		}
#endif
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
					float(texture.width * texture.height) * abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
				const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
				tri.LOD = max(0.f, sqrt(0.5f * log2f(Ta / Pa)));
			}
		}
	}
	else
	{
		for (const auto &mesh : meshes)
		{
			if (mesh.flags & SceneMesh::HAS_INDICES)
			{
				for (int i = 0, s = static_cast<int>(mesh.faceCount); i < s; i++)
				{
					const auto triIdx = mesh.triangleOffset + i;
					const auto index = indices[i + mesh.faceOffset] + mesh.vertexOffset;
					Triangle &tri = triangles.at(triIdx);

					if (!texCoords.empty())
					{
						tri.u0 = texCoords[index.x].x;
						tri.v0 = texCoords[index.x].y;

						tri.u1 = texCoords[index.y].x;
						tri.v1 = texCoords[index.y].y;

						tri.u2 = texCoords[index.z].x;
						tri.v2 = texCoords[index.z].y;
					}

					tri.material = this->materialIndices[triIdx];

					const HostMaterial &mat = matList->get(tri.material);
					int texID = mat.map[0].textureID;
					if (texID > -1)
					{
						const Texture &texture = matList->getTextures().at(texID);

						const float Ta = static_cast<float>(texture.width * texture.height) *
										 abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
						const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
						tri.LOD = 0.5f * log2f(Ta / Pa);
					}
				}
			}
			else
			{
				for (int i = 0, s = static_cast<int>(mesh.faceCount); i < s; i++)
				{
					const auto triIdx = mesh.triangleOffset + i;
					const auto idx = i * 3;
					const uvec3 index = uvec3(idx + 0, idx + 1, idx + 2) + mesh.vertexOffset;
					Triangle &tri = triangles.at(triIdx);

					if (!texCoords.empty())
					{
						tri.u0 = texCoords[index.x].x;
						tri.v0 = texCoords[index.x].y;

						tri.u1 = texCoords[index.y].x;
						tri.v1 = texCoords[index.y].y;

						tri.u2 = texCoords[index.z].x;
						tri.v2 = texCoords[index.z].y;
					}

					tri.material = this->materialIndices[triIdx];

					const HostMaterial &mat = matList->get(tri.material);
					int texID = mat.map[0].textureID;
					if (texID > -1)
					{
						const Texture &texture = matList->getTextures()[texID];

						const float Ta = static_cast<float>(texture.width * texture.height) *
										 abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
						const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
						tri.LOD = 0.5f * log2f(Ta / Pa);
					}
				}
			}
		}
	}
}
