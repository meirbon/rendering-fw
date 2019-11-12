#ifndef RENDERINGFW_VULKANRTX_SRC_MESH_H
#define RENDERINGFW_VULKANRTX_SRC_MESH_H

#include <vulkan/vulkan.hpp>
#include <MathIncludes.h>
#include <Structures.h>
#include <DeviceStructures.h>

#include "VulkanDevice.h"
#include "VmaBuffer.h"
#include "AccelerationStructure.h"

#include <vk_mem_alloc.h>

namespace vkrtx
{

class Mesh
{
  public:
	Mesh(const VulkanDevice &device);
	~Mesh();

	void cleanup();
	void setGeometry(const rfw::Mesh &mesh);

	VmaBuffer<rfw::DeviceTriangle> triangles;
	BottomLevelAS *accelerationStructure = nullptr;

  private:
	VulkanDevice m_Device;
};

} // namespace vkrtx
#endif // RENDERINGFW_VULKANRTX_SRC_MESH_H
