#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <optix.h>
#include <optix_world.h>

#include "CUDABuffer.h"
#include "OptiXCUDABuffer.h"
#include <Structures.h>

class OptiXMesh
{
  public:
	OptiXMesh(optix::Context &context, optix::Program attribProgram);
	~OptiXMesh();

	void cleanup();

	void setData(const rfw::Mesh &mesh);

	optix::GeometryTriangles optixTriangles;
	CUDABuffer<rfw::Triangle> *triangles = nullptr;

  private:
	OptiXCUDABuffer<glm::vec4, OptiXBufferType::Read> *m_Vertices = nullptr;
	OptiXCUDABuffer<glm::uvec3, OptiXBufferType::Read> *m_Indices = nullptr;

	uint vertexCount = 0;
	uint triangleCount = 0;
	optix::Context m_Context;
	optix::Program m_AttribProgram;
};