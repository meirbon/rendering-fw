#include "Context.h"

using namespace vkrtx;

BottomLevelAS::BottomLevelAS(VulkanDevice device, const glm::vec4 *vertices, uint32_t vertexCount,
							 const glm::uvec3 *indices, uint32_t indexCount, AccelerationStructureType type)
	: m_Device(device), m_Type(type), m_Vertices(device), m_Indices(device), m_Memory(device)
{
	assert(vertexCount > 0);
	m_Vertices.allocate(
		vertexCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
		vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_Vertices.copyToDevice(vertices, vertexCount * sizeof(glm::vec4));

	if (indexCount > 0)
	{
		m_Indices.allocate(indexCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
						   vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst,
						   VMA_MEMORY_USAGE_GPU_ONLY);
		m_Indices.copyToDevice(indices, indexCount * sizeof(glm::uvec3));
	}

	m_Flags = typeToFlags(type);

	m_Geometry.pNext = nullptr;
	m_Geometry.geometryType = vk::GeometryTypeNV::eTriangles;
	m_Geometry.geometry.triangles.pNext = nullptr;
	m_Geometry.geometry.triangles.vertexData = static_cast<vk::Buffer>(m_Vertices);
	m_Geometry.geometry.triangles.vertexOffset = 0;
	m_Geometry.geometry.triangles.vertexCount = vertexCount;
	m_Geometry.geometry.triangles.vertexStride = sizeof(glm::vec4);
	m_Geometry.geometry.triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;

	if (indexCount > 0)
	{
		m_Geometry.geometry.triangles.indexData = static_cast<vk::Buffer>(m_Indices);
		m_Geometry.geometry.triangles.indexOffset = 0;
		m_Geometry.geometry.triangles.indexCount = indexCount * 3;
		m_Geometry.geometry.triangles.indexType = vk::IndexType::eUint32;
	}
	else
	{
		m_Geometry.geometry.triangles.indexData = nullptr;
		m_Geometry.geometry.triangles.indexOffset = 0;
		m_Geometry.geometry.triangles.indexCount = 0;
		m_Geometry.geometry.triangles.indexType = vk::IndexType::eNoneNV;
	}

	m_Geometry.geometry.triangles.transformData = nullptr;
	m_Geometry.geometry.triangles.transformOffset = 0;
	m_Geometry.flags = vk::GeometryFlagBitsNV::eOpaque;

	// Create the descriptor of the acceleration structure,
	// which contains the number of geometry descriptors it will contain
	vk::AccelerationStructureInfoNV accelerationStructureInfo = {vk::AccelerationStructureTypeNV::eBottomLevel, m_Flags,
																 0, 1, &m_Geometry};

	vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {0, accelerationStructureInfo};

	CheckVK(device->createAccelerationStructureNV(&accelerationStructureCreateInfo, nullptr, &m_Structure,
												  m_Device.getLoader()));

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eObject;

	vk::MemoryRequirements2 memoryRequirements;
	device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														 m_Device.getLoader());

	const auto requirements = memoryRequirements.memoryRequirements;

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch;
	device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														 m_Device.getLoader());
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eUpdateScratch;
	device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														 m_Device.getLoader());
	m_ScratchSize = std::max(m_ScratchSize, memoryRequirements.memoryRequirements.size);

// Create result memory
#if 1
	m_Memory.allocate(m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
					  vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferSrc, ON_DEVICE);
#else
	m_Memory.allocate(m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
					  vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferSrc,
					  VMA_MEMORY_USAGE_GPU_ONLY, true, requirements);
#endif

	// bind the acceleration structure descriptor to the actual memory that will contain it
	VkBindAccelerationStructureMemoryInfoNV bindInfo{};
	bindInfo.accelerationStructure = m_Structure;
	bindInfo.memory = m_Memory.getDeviceMemory();
	bindInfo.deviceIndexCount = 0;
	bindInfo.memoryOffset = 0;
	bindInfo.pDeviceIndices = nullptr;
	device.getLoader().vkBindAccelerationStructureMemoryNV(device, 1, &bindInfo);
}

BottomLevelAS::~BottomLevelAS() { cleanup(); }

void BottomLevelAS::cleanup()
{
	if (m_Structure)
	{
		m_Device.getLoader().vkDestroyAccelerationStructureNV(m_Device, m_Structure, nullptr);
		m_Structure = nullptr;
	}

	m_Vertices.cleanup();
	m_Memory.cleanup();
}

void BottomLevelAS::updateVertices(const glm::vec4 *vertices, uint32_t vertexCount)
{
	assert(m_Vertices.getElementCount() == vertexCount);
	m_Vertices.copyToDevice(vertices, vertexCount * sizeof(glm::vec4));
}

void BottomLevelAS::build(const VmaBuffer<uint8_t> &scratchBuffer) { build(false, scratchBuffer); }

void BottomLevelAS::rebuild(const VmaBuffer<uint8_t> &scratchBuffer) { build(true, scratchBuffer); }

uint64_t BottomLevelAS::getHandle()
{
	uint64_t handle = 0;
	CheckVK(m_Device->getAccelerationStructureHandleNV(m_Structure, sizeof(uint64_t), &handle, m_Device.getLoader()));
	assert(handle);
	return handle;
}

uint32_t BottomLevelAS::getVertexCount() const { return static_cast<uint32_t>(m_Vertices.getElementCount()); }

void BottomLevelAS::build(bool update, VmaBuffer<uint8_t> scratchBuffer)
{
	assert(m_Vertices.getElementCount() > 0);

	// Create temporary scratch buffer
	//	auto scratchBuffer = VmaBuffer<uint8_t>(m_Device, m_ScratchSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
	//											vk::BufferUsageFlagBits::eRayTracingNV, VMA_MEMORY_USAGE_GPU_ONLY);

	// Won't reallocate if buffer is big enough
	scratchBuffer.reallocate(m_ScratchSize);

	// build the actual bottom-level AS
	vk::AccelerationStructureInfoNV buildInfo = {vk::AccelerationStructureTypeNV::eBottomLevel, m_Flags, 0, 1,
												 &m_Geometry};

	// submit build command
	auto commandBuffer = m_Device.createOneTimeCmdBuffer();
	auto computeQueue = m_Device.getComputeQueue();

	// Never compact BVHs that are supposed to be updated
	if (!update && (m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction))
	{
		commandBuffer->buildAccelerationStructureNV(&buildInfo, nullptr, 0, update, m_Structure, nullptr,
													static_cast<vk::Buffer>(scratchBuffer), 0, m_Device.getLoader());
		// Create memory barrier for building AS to make sure it can only be used when ready
		vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV |
											   vk::AccessFlagBits::eAccelerationStructureReadNV,
										   vk::AccessFlagBits::eAccelerationStructureReadNV};
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
									   vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1,
									   &memoryBarrier, 0, nullptr, 0, nullptr);

		// Create query pool to get compacted AS size
		vk::QueryPoolCreateInfo queryPoolCreateInfo = vk::QueryPoolCreateInfo(
			vk::QueryPoolCreateFlags(), vk::QueryType::eAccelerationStructureCompactedSizeNV, 1);
		vk::QueryPool queryPool = m_Device->createQueryPool(queryPoolCreateInfo);

		// Query for compacted size
		commandBuffer->resetQueryPool(queryPool, 0, 1);
		commandBuffer->beginQuery(queryPool, 0, vk::QueryControlFlags());
		m_Device.getLoader().vkCmdWriteAccelerationStructuresPropertiesNV(
			commandBuffer.getVkCommandBuffer(), 1, (VkAccelerationStructureNV *)(&m_Structure),
			VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_NV, queryPool, 0);

		commandBuffer->endQuery(queryPool, 0);
		commandBuffer.submit(computeQueue, true);

		uint32_t size = 0;
		CheckVK(m_Device->getQueryPoolResults(queryPool, 0, 1, sizeof(uint32_t), &size, sizeof(uint32_t),
											  vk::QueryResultFlagBits::eWait));

		if (size > 0) // Only compact if the queried result returns a valid size value
		{
			buildInfo.geometryCount = 0; // Must be zero for compacted AS
			buildInfo.pGeometries = nullptr;
			vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo = {0, buildInfo};
			accelerationStructureCreateInfo.compactedSize = size;
			// Create AS handle
			vk::AccelerationStructureNV compactedAS;

			CheckVK(m_Device.getLoader().vkCreateAccelerationStructureNV(
				m_Device, (const VkAccelerationStructureCreateInfoNV *)&accelerationStructureCreateInfo, nullptr,
				(VkAccelerationStructureNV *)(&compactedAS)));
			// Get new memory requirements
			vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo = {
				vk::AccelerationStructureMemoryRequirementsTypeNV::eObject, compactedAS};
			vk::MemoryRequirements2 memoryRequirements;
			m_Device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
																   m_Device.getLoader());
#if 1
			// Create new, smaller buffer for compacted AS
			auto newMemory = Buffer<uint8_t>(
				m_Device, memoryRequirements.memoryRequirements.size, vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst, ON_DEVICE);
#else
			auto newMemory = VmaBuffer<uint8_t>(
				m_Device, memoryRequirements.memoryRequirements.size, vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, true);
#endif
			// bind the acceleration structure descriptor to the memory that will contain it
			vk::BindAccelerationStructureMemoryInfoNV bindInfo = {compactedAS, newMemory, 0, 0, nullptr};
			CheckVK(m_Device.getLoader().vkBindAccelerationStructureMemoryNV(
				m_Device, 1, (const VkBindAccelerationStructureMemoryInfoNV *)&bindInfo));

			// submit copy & compact command to command buffer
			commandBuffer.begin();
			commandBuffer->copyAccelerationStructureNV(
				compactedAS, m_Structure, vk::CopyAccelerationStructureModeNV::eCompact, m_Device.getLoader());
			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
										   vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1,
										   &memoryBarrier, 0, nullptr, 0, nullptr);
			commandBuffer.submit(computeQueue, true);

			// cleanup
			m_Device->destroyQueryPool(queryPool);
			m_Device.getLoader().vkDestroyAccelerationStructureNV(m_Device, m_Structure, nullptr);
			m_Memory.cleanup();

			// Assign new AS to this object
			m_Memory = newMemory;
			m_Structure = compactedAS;
		}
	}
	else
	{
		commandBuffer->buildAccelerationStructureNV(&buildInfo, nullptr, 0, update, m_Structure,
													update ? m_Structure : nullptr,
													static_cast<vk::Buffer>(scratchBuffer), 0, m_Device.getLoader());
		// Create memory barrier for building AS to make sure it can only be used when ready
		vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV |
											   vk::AccessFlagBits::eAccelerationStructureReadNV,
										   vk::AccessFlagBits::eAccelerationStructureReadNV};
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
									   vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1,
									   &memoryBarrier, 0, nullptr, 0, nullptr);
		commandBuffer.submit(computeQueue, true);
	}
}
