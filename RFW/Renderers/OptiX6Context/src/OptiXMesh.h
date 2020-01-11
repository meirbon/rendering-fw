#pragma once

#define NOMINMAX
#include <optix.h>
#include <optix_world.h>

#include "CUDABuffer.h"
#include <Structures.h>

class OptiXMesh
{
  public:
	OptiXMesh(optix::Context &context, optix::Program attribProgram);
	~OptiXMesh();

	void cleanup();

	void setData(const rfw::Mesh &mesh, optix::Material material);

	CUDABuffer<rfw::Triangle> &getTrianglesBuffer() { return *m_Triangles; }

	[[nodiscard]] optix::GeometryTriangles getGeometryTriangles() const { return m_OptiXTriangles; }
	[[nodiscard]] optix::Acceleration getAcceleration() const { return m_Acceleration; }
  private:
	CUDABuffer<rfw::Triangle> *m_Triangles = nullptr;

	uint vertexCount = 0;
	uint triangleCount = 0;
	optix::Context m_Context;
	optix::Buffer m_VertexBuffer;
	optix::Buffer m_IndexBuffer;
	optix::Program m_AttribProgram;
	optix::GeometryTriangles m_OptiXTriangles;
	optix::Acceleration m_Acceleration;
};