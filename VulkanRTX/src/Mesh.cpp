#include "Mesh.h"

using namespace vkrtx;

Mesh::Mesh(const VulkanDevice &device)
	: m_Device(device), triangles(device)
{
	// Make sure at least a buffer exists
	triangles.allocate(1, vk::MemoryPropertyFlagBits::eDeviceLocal,
					   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
					   VMA_MEMORY_USAGE_GPU_ONLY);
}

Mesh::~Mesh() { cleanup(); }

void Mesh::setGeometry(const rfw::Mesh &mesh)
{
	const bool sameTriCount = triangles && (triangles.getSize() / sizeof(rfw::DeviceTriangle) == mesh.triangleCount);

	if (triangles.getSize() == 1) // Initialize buffer with settings optimized for static geometry
	{
		triangles.allocate(mesh.triangleCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
						   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
						   VMA_MEMORY_USAGE_GPU_ONLY);
	}
	else if (!sameTriCount) // Reinitialize buffer with settings optimized for frequently updated geometry
	{
		triangles.cleanup();
		triangles.allocate(mesh.triangleCount,
						   vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
						   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst,
						   VMA_MEMORY_USAGE_CPU_TO_GPU);
	}

	triangles.copyToDevice(mesh.triangles, mesh.triangleCount * sizeof(rfw::DeviceTriangle));

	if (accelerationStructure != nullptr)
	{
		if (accelerationStructure->canUpdate() && sameTriCount)
		{
			// Same data count, rebuild acceleration structure
			accelerationStructure->updateVertices(mesh.vertices, uint32_t(mesh.vertexCount));
			accelerationStructure->rebuild();
		}
		else
		{
			// Create new, updateable acceleration structure
			delete accelerationStructure;
			accelerationStructure = nullptr;
			accelerationStructure =
				new BottomLevelAS(m_Device, mesh.vertices, uint32_t(mesh.vertexCount), mesh.indices,
								  mesh.indices ? uint32_t(mesh.triangleCount) : 0, FastTrace);
			accelerationStructure->build();
		}
	}
	else
	{
		// Create initial acceleration structure
		accelerationStructure =
			new BottomLevelAS(m_Device, mesh.vertices, uint32_t(mesh.vertexCount), mesh.indices,
							  mesh.indices ? uint32_t(mesh.triangleCount) : 0, FastestTrace);
		accelerationStructure->build();
	}

	assert(accelerationStructure);
}

void Mesh::cleanup()
{
	if (accelerationStructure)
		delete accelerationStructure, accelerationStructure = nullptr;
	triangles.cleanup();
}
