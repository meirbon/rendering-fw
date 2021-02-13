#pragma once

#include "VmaBuffer.h"

#include <rfw/math.h>
#include <rfw/context/device_structs.h>

#include "VulkanDevice.h"

namespace vkc
{

class VkMesh
{
  public:
	VkMesh(const VulkanDevice &device);
	~VkMesh();

	void setGeometry(const rfw::Mesh &mesh);

	[[nodiscard]] bool hasIndices() const;

	uint vertexCount = 0;
	uint triangleCount = 0;
	VmaBuffer<glm::vec3> vertices;
	VmaBuffer<glm::vec3> normals;
	VmaBuffer<glm::uvec3> indices;
	VmaBuffer<rfw::DeviceTriangle> triangles;

  private:
	vkc::VulkanDevice m_Device;
};

} // namespace vkc