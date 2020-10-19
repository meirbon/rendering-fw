#include <rfw/context/structs.h>
#include "VkMesh.h"

vkc::VkMesh::VkMesh(const vkc::VulkanDevice &device) : m_Device(device), vertices(device), normals(device), triangles(device), indices(device) {}

vkc::VkMesh::~VkMesh() {}

void vkc::VkMesh::setGeometry(const rfw::Mesh &mesh)
{
	vertices.allocate(mesh.vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
					  vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	vertices.copyToDevice(mesh.vertices);

	normals.allocate(mesh.vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
					 vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	normals.copyToDevice(mesh.normals);

	if (mesh.hasIndices())
	{
		indices.allocate(mesh.triangleCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
						 vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);
		indices.copyToDevice(mesh.indices);
	}

	triangles.allocate(mesh.triangleCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
					   vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	triangles.copyToDevice(mesh.triangles);
}

bool vkc::VkMesh::hasIndices() const { return indices.getSize() > 0; }
