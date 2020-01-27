#include "PCH.h"

using namespace rfw;

CPUMesh::~CPUMesh()
{
	if (scene)
		rtcReleaseScene(scene);

	scene = nullptr;
}

CPUMesh::CPUMesh(const CPUMesh &other)
{
	memcpy(this, &other, sizeof(CPUMesh));
	memset(const_cast<CPUMesh *>(&other), 0, sizeof(CPUMesh));
}

void CPUMesh::setGeometry(const Mesh &mesh)
{
	if (vertices)
	{
		vertices = mesh.vertices;
		triangles = mesh.triangles;

		auto geometry = rtcGetGeometry(scene, ID);
		vertexCount = int(mesh.vertexCount);
		triangleCount = int(mesh.triangleCount);
		rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, mesh.vertices, 0, sizeof(vec4), mesh.vertexCount);
		// rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_NORMAL, 1, RTC_FORMAT_FLOAT3, mesh.normals, 0, sizeof(vec3), mesh.vertexCount);
		if (mesh.hasIndices())
			rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh.indices, 0, sizeof(uvec3), mesh.triangleCount);
		if (mesh.vertexCount == vertexCount) // Animated data
			rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_REFIT);
		rtcCommitGeometry(geometry);
	}
	else
	{
		vertices = mesh.vertices;
		triangles = mesh.triangles;
		vertexCount = int(mesh.vertexCount);
		triangleCount = int(mesh.triangleCount);

		auto geometry = rtcNewGeometry(device, RTC_GEOMETRY_TYPE_TRIANGLE);
		rtcSetGeometryVertexAttributeCount(geometry, 1);
		rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, mesh.vertices, 0, sizeof(vec4), mesh.vertexCount);
		// rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_NORMAL, 1, RTC_FORMAT_FLOAT3, mesh.normals, 0, sizeof(vec3), mesh.vertexCount);
		if (mesh.hasIndices())
			rtcSetSharedGeometryBuffer(geometry, RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, mesh.indices, 0, sizeof(uvec3), mesh.triangleCount);
		rtcSetGeometryBuildQuality(geometry, RTC_BUILD_QUALITY_HIGH);
		rtcCommitGeometry(geometry);

		scene = rtcNewScene(device);
		ID = rtcAttachGeometry(scene, geometry);
	}

	rtcCommitScene(scene);
}
