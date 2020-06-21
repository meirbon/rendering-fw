#include "Context.h"

using namespace vkrtx;

BottomLevelAS::BottomLevelAS(VulkanDevice device, const glm::vec4 *vertices, uint32_t vertexCount,
							 const glm::uvec3 *indices, uint32_t indexCount, AccelerationStructureType type)
	: m_Device(device), m_Type(type), m_Vertices(device), m_Indices(device), m_Memory(device)
{
	assert(vertexCount > 0);
	m_Vertices.allocate(vertexCount,
						vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
						vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferDst |
							vk::BufferUsageFlagBits::eShaderDeviceAddressKHR,
						VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_Vertices.copyToDevice(vertices, vertexCount * sizeof(glm::vec4));

	if (indexCount > 0)
	{
		m_Indices.allocate(indexCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
						   vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferDst,
						   VMA_MEMORY_USAGE_GPU_ONLY);
		m_Indices.copyToDevice(indices, indexCount * sizeof(glm::uvec3));
	}

	m_Flags = typeToFlags(type);

	m_GeometryTriangles.indexType = vk::IndexType::eUint32;
	m_GeometryTriangles.setIndexType(vk::IndexType::eUint32);
	m_GeometryTriangles.setVertexData(m_Vertices.get_buffer_address());
	m_GeometryTriangles.setVertexStride(sizeof(vec4));
	m_GeometryTriangles.setTransformData({});

	m_GeometryType = vk::GeometryTypeKHR::eTriangles;
	if (indexCount > 0)
	{
		m_PrimCount = indexCount;
		m_GeometryTriangles.setIndexData(m_Indices.get_buffer_address());
	}
	else
	{
		m_PrimCount = vertexCount / 3;
		m_GeometryTriangles.indexData = nullptr;
	}

	/*m_Geometry.geometry.triangles.transformData = nullptr;
	m_Geometry.geometry.triangles.transformOffset = 0;
	m_Geometry.flags = vk::GeometryFlagBitsKHR::eOpaque;*/

	m_Geometry.setGeometryType(vk::GeometryTypeKHR::eTriangles);
	m_Geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);
	m_Geometry.geometry.setTriangles(m_GeometryTriangles);

	m_Offset.setFirstVertex(0);
	m_Offset.setPrimitiveCount(m_PrimCount);
	m_Offset.setPrimitiveOffset(0);
	m_Offset.setTransformOffset(0);

	// Create the descriptor of the acceleration structure,
	// which contains the number of geometry descriptors it will contain
	m_GeometryTypeInfo = vk::AccelerationStructureCreateGeometryTypeInfoKHR(
		m_GeometryType, m_PrimCount, vk::IndexType::eUint32, vertexCount, vk::Format::eR32G32B32Sfloat, false);
	vk::AccelerationStructureCreateInfoKHR accelerationStructureInfo = {
		0, vk::AccelerationStructureTypeKHR::eBottomLevel, m_Flags, 1, &m_GeometryTypeInfo};

	m_Structure = device->createAccelerationStructureKHR(accelerationStructureInfo, nullptr, m_Device.getLoader());

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoKHR memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eObject;
	memoryRequirementsInfo.buildType = vk::AccelerationStructureBuildTypeKHR::eHostOrDevice;

	vk::MemoryRequirements2 memoryRequirements;
	device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
														  m_Device.getLoader());

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eBuildScratch;
	device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
														  m_Device.getLoader());
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eUpdateScratch;
	device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
														  m_Device.getLoader());
	m_ScratchSize = std::max(m_ScratchSize, memoryRequirements.memoryRequirements.size);

// Create result memory
#if 1
	m_Memory.allocate(m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
					  vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferSrc, ON_DEVICE);
#else
	m_Memory.allocate(m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
					  vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferSrc,
					  VMA_MEMORY_USAGE_GPU_ONLY, true, requirements);
#endif

	// bind the acceleration structure descriptor to the actual memory that will contain it
	vk::BindAccelerationStructureMemoryInfoKHR bindInfo{};
	bindInfo.accelerationStructure = m_Structure;
	bindInfo.memory = m_Memory.getDeviceMemory();
	bindInfo.deviceIndexCount = 0;
	bindInfo.memoryOffset = 0;
	bindInfo.pDeviceIndices = nullptr;
	m_Device->bindAccelerationStructureMemoryKHR({bindInfo}, m_Device.getLoader());
}

BottomLevelAS::~BottomLevelAS() { cleanup(); }

void BottomLevelAS::cleanup()
{
	if (m_Structure)
	{
		m_Device.getLoader().vkDestroyAccelerationStructureKHR(m_Device, m_Structure, nullptr);
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

vk::DeviceAddress BottomLevelAS::getHandle()
{
	vk::AccelerationStructureDeviceAddressInfoKHR info = {m_Structure};
	return m_Device->getAccelerationStructureAddressKHR(info, m_Device.getLoader());
}

uint32_t BottomLevelAS::getVertexCount() const { return static_cast<uint32_t>(m_Vertices.getElementCount()); }

void BottomLevelAS::build(bool update, VmaBuffer<uint8_t> scratchBuffer)
{
	assert(m_Vertices.getElementCount() > 0);

	// Create temporary scratch buffer
	//	auto scratchBuffer = VmaBuffer<uint8_t>(m_Device, m_ScratchSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
	//											vk::BufferUsageFlagBits::eRayTracingKHR, VMA_MEMORY_USAGE_GPU_ONLY);

	// Won't reallocate if buffer is big enough
	scratchBuffer.reallocate(m_ScratchSize);

	// build the actual bottom-level AS
	const vk::AccelerationStructureGeometryKHR *pGeometry = &m_Geometry;
	vk::AccelerationStructureBuildGeometryInfoKHR buildInfo{};
	buildInfo.setFlags(m_Flags);
	buildInfo.setUpdate(false);
	buildInfo.setSrcAccelerationStructure({});
	buildInfo.setDstAccelerationStructure(m_Structure);
	buildInfo.setGeometryArrayOfPointers(false);
	buildInfo.setGeometryCount(1);
	buildInfo.setPpGeometries(&pGeometry);
	buildInfo.setScratchData(scratchBuffer.get_buffer_address());

	// submit build command
	auto commandBuffer = m_Device.createOneTimeCmdBuffer();
	auto computeQueue = m_Device.getComputeQueue();

	// Never compact BVHs that are supposed to be updated
	if (!update && (m_Flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction))
	{
		commandBuffer->buildAccelerationStructureKHR({buildInfo}, {&m_Offset}, m_Device.getLoader());

		// Create memory barrier for building AS to make sure it can only be used when ready
		vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteKHR |
											   vk::AccessFlagBits::eAccelerationStructureReadKHR,
										   vk::AccessFlagBits::eAccelerationStructureReadKHR};
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
									   vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::DependencyFlags(), 1,
									   &memoryBarrier, 0, nullptr, 0, nullptr);

		// Create query pool to get compacted AS size
		vk::QueryPoolCreateInfo queryPoolCreateInfo = vk::QueryPoolCreateInfo(
			vk::QueryPoolCreateFlags(), vk::QueryType::eAccelerationStructureCompactedSizeKHR, 1);
		vk::QueryPool queryPool = m_Device->createQueryPool(queryPoolCreateInfo);

		// Query for compacted size
		commandBuffer->resetQueryPool(queryPool, 0, 1);
		commandBuffer->beginQuery(queryPool, 0, vk::QueryControlFlags());
		m_Device.getLoader().vkCmdWriteAccelerationStructuresPropertiesKHR(
			commandBuffer.getVkCommandBuffer(), 1, (const VkAccelerationStructureKHR *)(&m_Structure),
			VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR, queryPool, 0);

		commandBuffer->endQuery(queryPool, 0);
		commandBuffer.submit(computeQueue, true);

		uint32_t size = 0;
		CheckVK(m_Device->getQueryPoolResults(queryPool, 0, 1, sizeof(uint32_t), &size, sizeof(uint32_t),
											  vk::QueryResultFlagBits::eWait));

		if (size > 0) // Only compact if the queried result returns a valid size value
		{
			buildInfo.geometryCount = 0; // Must be zero for compacted AS
			buildInfo.ppGeometries = nullptr;
			vk::AccelerationStructureCreateInfoKHR accelerationStructureCreateInfo = {
				size, vk::AccelerationStructureTypeKHR::eBottomLevel, m_Flags, 1, &m_GeometryTypeInfo};
			accelerationStructureCreateInfo.compactedSize = size;
			vk::AccelerationStructureCreateInfoKHR accelerationStructureInfo = {
				0, vk::AccelerationStructureTypeKHR::eBottomLevel, m_Flags, 1, &m_GeometryTypeInfo};

			vk::AccelerationStructureKHR compactedAS =
				m_Device->createAccelerationStructureKHR(accelerationStructureInfo, nullptr, m_Device.getLoader());

			buildInfo.srcAccelerationStructure = m_Structure;
			buildInfo.dstAccelerationStructure = compactedAS;

			// Get new memory requirements
			vk::AccelerationStructureMemoryRequirementsInfoKHR memoryRequirementsInfo = {
				vk::AccelerationStructureMemoryRequirementsTypeKHR::eObject,
				vk::AccelerationStructureBuildTypeKHR::eHostOrDevice, compactedAS};
			vk::MemoryRequirements2 memoryRequirements;
			m_Device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
																	m_Device.getLoader());
#if 1
			// Create new, smaller buffer for compacted AS
			auto newMemory = Buffer<uint8_t>(
				m_Device, memoryRequirements.memoryRequirements.size, vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferDst, ON_DEVICE);
#else
			auto newMemory = VmaBuffer<uint8_t>(
				m_Device, memoryRequirements.memoryRequirements.size, vk::MemoryPropertyFlagBits::eDeviceLocal,
				vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferDst,
				VMA_MEMORY_USAGE_GPU_ONLY, true);
#endif
			// bind the acceleration structure descriptor to the memory that will contain it
			vk::BindAccelerationStructureMemoryInfoKHR bindInfo = {compactedAS, newMemory, 0, 0, nullptr};
			CheckVK(m_Device.getLoader().vkBindAccelerationStructureMemoryKHR(
				m_Device, 1, (const VkBindAccelerationStructureMemoryInfoKHR *)&bindInfo));

			auto copyInfo = vk::CopyAccelerationStructureInfoKHR{m_Structure, compactedAS,
																 vk::CopyAccelerationStructureModeKHR::eCompact};
			// submit copy & compact command to command buffer
			commandBuffer.begin();

			commandBuffer->copyAccelerationStructureKHR(copyInfo, m_Device.getLoader());
			commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
										   vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::DependencyFlags(), 1,
										   &memoryBarrier, 0, nullptr, 0, nullptr);
			commandBuffer.submit(computeQueue, true);

			// cleanup
			m_Device->destroyQueryPool(queryPool);
			m_Device.getLoader().vkDestroyAccelerationStructureKHR(m_Device, m_Structure, nullptr);
			m_Memory.cleanup();

			// Assign new AS to this object
			m_Memory = newMemory;
			m_Structure = compactedAS;
		}
	}
	else
	{
		commandBuffer->buildAccelerationStructureKHR({buildInfo}, {&m_Offset}, m_Device.getLoader());

		// Create memory barrier for building AS to make sure it can only be used when ready
		auto memoryBarrier = vk::MemoryBarrier{vk::AccessFlagBits::eAccelerationStructureWriteKHR |
												   vk::AccessFlagBits::eAccelerationStructureReadKHR,
											   vk::AccessFlagBits::eAccelerationStructureReadKHR};
		commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
									   vk::PipelineStageFlagBits::eRayTracingShaderKHR, vk::DependencyFlags(), 1,
									   &memoryBarrier, 0, nullptr, 0, nullptr);
		commandBuffer.submit(computeQueue, true);
	}
}
