#pragma once

#include "PCH.h"

namespace rfw
{
class CPUMesh
{
  public:
	explicit CPUMesh(RTCDevice dev) : device(dev) {}
	~CPUMesh();
	CPUMesh(const CPUMesh &other);

	void setGeometry(const Mesh &mesh);

	const glm::vec4 *vertices = nullptr;
	const rfw::Triangle *triangles = nullptr;

	glm::vec4 *embreeVertices = nullptr;

	uint ID = 0;

	RTCDevice device = nullptr;
	RTCScene scene = nullptr;

  private:
	int vertexCount = 0;
	int triangleCount = 0;
};
} // namespace rfw