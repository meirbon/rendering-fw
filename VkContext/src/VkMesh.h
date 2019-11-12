#pragma once

#include "Buffer.h"

#include <MathIncludes.h>
#include <DeviceStructures.h>

namespace vkc
{

class VkMesh
{
  public:
	VkMesh(const Device& device);
	~VkMesh();

	void setGeometry(const rfw::Mesh& mesh);

	[[nodiscard]] bool hasIndices() const;

	uint vertexCount = 0;
	uint triangleCount = 0;
	Buffer<glm::vec3> *vertices = nullptr;
	Buffer<glm::vec3> *normals = nullptr;
	Buffer<glm::uvec3> *indices = nullptr;
	Buffer<rfw::DeviceTriangle> *triangles = nullptr;
  private:
	vkc::Device m_Device;
};

} // namespace vkc