#pragma once

#include <Structures.h>

#include <embree3/rtcore.h>
#include <embree3/rtcore_scene.h>
#include <embree3/rtcore_builder.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_buffer.h>

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