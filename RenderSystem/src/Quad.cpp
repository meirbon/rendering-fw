#include "Quad.h"

#include "RenderSystem.h"

rfw::Quad::Quad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, const uint material) : m_MatID(material)
{
	const vec3 normal = normalize(N); // let's not assume the normal is normalized.
	const vec3 tmp = N.x > 0.9f ? vec3(0, 1, 0) : vec3(1, 0, 0);
	const vec3 T = 0.5f * width * normalize(cross(N, tmp));
	const vec3 B = 0.5f * height * normalize(cross(normalize(T), N));

	m_Vertices.resize(6);
	m_Vertices.at(0) = vec4(pos - B - T, 1.0f);
	m_Vertices.at(1) = vec4(pos + B - T, 1.0f);
	m_Vertices.at(2) = vec4(pos - B + T, 1.0f);
	m_Vertices.at(3) = vec4(pos + B - T, 1.0f);
	m_Vertices.at(4) = vec4(pos + B + T, 1.0f);
	m_Vertices.at(5) = vec4(pos - B + T, 1.0f);

	m_Normals.resize(6, N);

	m_Triangles.resize(2);
	Triangle tri1, tri2;
	tri1.material = material;
	tri2.material = material;
	tri1.vN0 = tri1.vN1 = tri1.vN2 = N;
	tri2.vN0 = tri2.vN1 = tri2.vN2 = N;
	tri1.Nx = N.x;
	tri1.Ny = N.y;
	tri1.Nz = N.z;
	tri2.Nx = N.x;
	tri2.Ny = N.y;
	tri2.Nz = N.z;
	tri1.u0 = tri1.u1 = tri1.u2 = tri1.v0 = tri1.v1 = tri1.v2 = 0;
	tri2.u0 = tri2.u1 = tri2.u2 = tri2.v0 = tri2.v1 = tri2.v2 = 0;
	tri1.vertex0 = vec3(m_Vertices[0]);
	tri1.vertex1 = vec3(m_Vertices[1]);
	tri1.vertex2 = vec3(m_Vertices[2]);
	tri2.vertex0 = vec3(m_Vertices[3]);
	tri2.vertex1 = vec3(m_Vertices[4]);
	tri2.vertex2 = vec3(m_Vertices[5]);
	m_Triangles.at(0) = tri1;
	m_Triangles.at(1) = tri2;

	m_MeshTransforms.resize(1, glm::mat4(1.0f));
}

void rfw::Quad::transformTo(float timeInSeconds) {}

const std::vector<std::vector<int>> &rfw::Quad::getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize)
{
	if (reinitialize)
	{
		m_LightIndices.resize(1);
		if (matLightFlags.at(m_MatID))
			m_LightIndices[0] = {0, 1};
		else
			m_LightIndices[0] = {};
	}

	return m_LightIndices;
}

const std::vector<std::pair<size_t, rfw::Mesh>> &rfw::Quad::getMeshes() const { return m_Meshes; }

const std::vector<glm::mat4> &rfw::Quad::getMeshTransforms() const { return m_MeshTransforms; }

void rfw::Quad::prepareMeshes(RenderSystem &rs)
{
	rfw::Mesh mesh;
	mesh.vertexCount = m_Vertices.size();
	mesh.vertices = m_Vertices.data();
	mesh.normals = m_Normals.data();
	mesh.triangles = m_Triangles.data();
	mesh.triangleCount = m_Triangles.size();
	m_Meshes.emplace_back(rs.requestMeshIndex(), mesh);
}
