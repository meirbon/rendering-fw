#include <Structures.h>
#include "VkMesh.h"

vkc::VkMesh::VkMesh(const vkc::Device &device) : m_Device(device) {}

vkc::VkMesh::~VkMesh()
{
	delete vertices;
	delete normals;
	delete indices;
	delete triangles;
	vertices = nullptr;
	normals = nullptr;
	indices = nullptr;
	triangles = nullptr;
}

void vkc::VkMesh::setGeometry(const rfw::Mesh &mesh)
{
	delete vertices;
	delete normals;
	delete indices;
	delete triangles;
	vertices = nullptr;
	normals = nullptr;
	indices = nullptr;
	triangles = nullptr;

	vertices = new Buffer<glm::vec3>(m_Device, mesh.vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
									 vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
									 ON_DEVICE);
	vertices->CopyToDevice(mesh.vertices);
	normals = new Buffer<glm::vec3>(m_Device, mesh.vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
									vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
									ON_DEVICE);
	normals->CopyToDevice(mesh.normals);

	if (mesh.hasIndices())
	{
		indices = new Buffer<glm::uvec3>(m_Device, mesh.triangleCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
										 vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eTransferDst,
										 ON_DEVICE);
		indices->CopyToDevice(mesh.indices);
	}

	triangles = new Buffer<rfw::DeviceTriangle>(
		m_Device, mesh.triangleCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, ON_DEVICE);
	triangles->CopyToDevice(mesh.triangles);
}

bool vkc::VkMesh::hasIndices() const { return indices != nullptr; }
