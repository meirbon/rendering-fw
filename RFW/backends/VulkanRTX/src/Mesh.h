#pragma once

#include "VulkanDevice.h"
#include "VmaBuffer.h"
#include "AccelerationStructure.h"

#include <rfw/context/structs.h>
#include <rfw/context/device_structs.h>

namespace vkrtx
{

class Mesh
{
  public:
	Mesh(const VulkanDevice &device);
	~Mesh();

	void cleanup();
	void setGeometry(const rfw::Mesh &mesh, const VmaBuffer<uint8_t> &scratchBuffer);

	VmaBuffer<rfw::DeviceTriangle> triangles;
	BottomLevelAS *accelerationStructure = nullptr;

  private:
	VulkanDevice m_Device;
};

} // namespace vkrtx
