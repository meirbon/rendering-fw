#include <MathIncludes.h>
#include "CUDABuffer.h"
#include "OptiXMesh.h"

#include <utils/Logger.h>

OptiXMesh::OptiXMesh(optix::Context &context, optix::Program attribProgram)
	: m_Context(context), m_AttribProgram(attribProgram)
{
	m_OptiXTriangles = m_Context->createGeometryTriangles();
	m_Acceleration = m_Context->createAcceleration("Trbvh");
	m_Acceleration->setProperty("refit", "1"); // Enable refitting
}

OptiXMesh::~OptiXMesh() { cleanup(); }

void OptiXMesh::cleanup()
{
	if (m_Triangles)
		delete m_Triangles;
	m_Triangles = nullptr;
}

void OptiXMesh::setData(const rfw::Mesh &mesh, optix::Material material)
{
	if (!m_Triangles || m_Triangles->getElementCount() < mesh.triangleCount)
	{
		delete m_Triangles;
		m_Triangles = new CUDABuffer<rfw::Triangle>(mesh.triangleCount, ON_DEVICE);
	}

	m_Triangles->copyToDevice(mesh.triangles, mesh.triangleCount);

	if (!m_VertexBuffer.get())
		m_VertexBuffer = m_Context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, mesh.vertexCount);
	else
	{
		RTsize size;
		m_VertexBuffer->getSize(size);
		if (size < mesh.vertexCount)
			m_VertexBuffer = m_Context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_FLOAT4, mesh.vertexCount);
	}

	if (mesh.hasIndices())
	{
		if (!m_IndexBuffer.get())
			m_IndexBuffer = m_Context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.triangleCount);
		else
		{
			RTsize size;
			m_IndexBuffer->getSize(size);
			if (size < mesh.triangleCount)
				m_IndexBuffer = m_Context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_INT3, mesh.triangleCount);
		}

		memcpy(m_IndexBuffer->map(), mesh.indices, mesh.triangleCount * sizeof(glm::uvec3));
		m_IndexBuffer->unmap();
		m_OptiXTriangles->setTriangleIndices(m_IndexBuffer, RT_FORMAT_UNSIGNED_INT3);
	}

	memcpy(m_VertexBuffer->map(), mesh.vertices, mesh.vertexCount * sizeof(glm::vec4));
	m_VertexBuffer->unmap();

	m_OptiXTriangles->setAttributeProgram(m_AttribProgram);
	m_OptiXTriangles->setPrimitiveCount(static_cast<uint>(mesh.triangleCount));
	m_OptiXTriangles->setVertices(static_cast<uint>(mesh.vertexCount), m_VertexBuffer, 0, sizeof(vec4), RT_FORMAT_FLOAT3);
	m_OptiXTriangles->setBuildFlags(RT_GEOMETRY_BUILD_FLAG_NONE);
#ifndef NDEBUG
	try
	{
		m_OptiXTriangles->validate();
	}
	catch (const std::exception &e)
	{
		WARNING(e.what());
	}
#else
	m_OptiXTriangles->validate();
#endif

	m_Acceleration->markDirty();
}
