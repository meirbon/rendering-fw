#include "OptiXContext.h"

OptiXMesh::OptiXMesh(optix::Context &context, optix::Program attribProgram) : m_Context(context), m_AttribProgram(attribProgram)
{
	optixTriangles = m_Context->createGeometryTriangles();
	optixTriangles->setAttributeProgram(m_AttribProgram);
}

OptiXMesh::~OptiXMesh() { cleanup(); }

void OptiXMesh::cleanup()
{
	if (triangles)
		delete triangles;
	triangles = nullptr;
}

void OptiXMesh::setData(const rfw::Mesh &mesh)
{
	const bool hasTriangles = triangles;

	if (!triangles || triangles->getElementCount() < mesh.triangleCount)
	{
		delete triangles;
		triangles = new CUDABuffer<rfw::Triangle>(mesh.triangleCount, ON_DEVICE);
	}

	triangles->copyToDeviceAsync(mesh.triangles, mesh.triangleCount);

	if (!m_Vertices || m_Vertices->size() < mesh.vertexCount)
	{
		delete m_Vertices;
		m_Vertices = new OptiXCUDABuffer<glm::vec4, Read>(m_Context, {mesh.vertexCount});
	}

	m_Vertices->copy_to_device(mesh.vertices, {mesh.vertexCount}, 0, true);
	assert(m_Vertices->size() == mesh.vertexCount);

	if (mesh.hasIndices() && (!m_Indices || triangleCount != mesh.triangleCount))
	{
		delete m_Indices;
		m_Indices = new OptiXCUDABuffer<glm::uvec3, Read>(m_Context, {mesh.triangleCount});
		m_Indices->copy_to_device(mesh.indices, {mesh.triangleCount}, 0, true);
		optixTriangles->setTriangleIndices(m_Indices->buffer(), RT_FORMAT_UNSIGNED_INT3);

		assert(m_Indices->size() == mesh.triangleCount);
	}

	optixTriangles->setPrimitiveCount(static_cast<uint>(mesh.triangleCount));
	optixTriangles->setVertices(static_cast<uint>(mesh.vertexCount), m_Vertices->buffer(), 0, sizeof(vec4), RT_FORMAT_FLOAT3);
	optixTriangles->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_NONE);
	try
	{
		optixTriangles->validate();
		CheckCUDA(cudaGetLastError());
	}
	catch (const std::exception &e)
	{
		WARNING("%s", e.what());
	}

	vertexCount = uint(mesh.vertexCount);
	triangleCount = uint(mesh.triangleCount);

	CheckCUDA(cudaDeviceSynchronize());
}
