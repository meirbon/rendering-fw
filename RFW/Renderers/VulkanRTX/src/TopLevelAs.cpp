#include "Context.h"

using namespace vkrtx;

TopLevelAS::TopLevelAS(const VulkanDevice &dev, AccelerationStructureType type, uint32_t instanceCount)
	: m_Device(dev), m_InstanceCnt(instanceCount), m_Type(type), m_Flags(typeToFlags(type)), m_Memory(dev),
	  m_InstanceBuffer(dev)
{
	m_InstanceBuffer.allocate(instanceCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
							  vk::BufferUsageFlagBits::eRayTracingKHR | vk::BufferUsageFlagBits::eTransferDst |
								  vk::BufferUsageFlagBits::eShaderDeviceAddressKHR);

	m_GeometryInstances.setPNext(nullptr);
	m_GeometryInstances.setArrayOfPointers(false);
	m_GeometryInstances.setData(m_InstanceBuffer.get_buffer_address());
	const auto *ptr = &m_GeometryInstances;

	m_GeometryTypeInfo = vk::AccelerationStructureCreateGeometryTypeInfoKHR(vk::GeometryTypeKHR::eInstances, 1);

	m_Geometry.setPNext(nullptr);
	m_Geometry.setGeometry(m_GeometryInstances);
	m_Geometry.setGeometryType(vk::GeometryTypeKHR::eInstances);
	m_Geometry.setFlags(vk::GeometryFlagBitsKHR::eOpaque);

	vk::AccelerationStructureCreateInfoKHR as_create_info{};
	as_create_info.pNext = nullptr;
	as_create_info.type = vk::AccelerationStructureTypeKHR::eTopLevel;
	as_create_info.flags = m_Flags;
	as_create_info.maxGeometryCount = 1;
	as_create_info.setPGeometryInfos(&m_GeometryTypeInfo);

	m_Offset.setFirstVertex(0);
	m_Offset.setPrimitiveCount(1);
	m_Offset.setPrimitiveOffset(0);
	m_Offset.setTransformOffset(0);

	m_Structure = m_Device->createAccelerationStructureKHR(as_create_info, nullptr, m_Device.getLoader());

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoKHR memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eObject;
	memoryRequirementsInfo.buildType = vk::AccelerationStructureBuildTypeKHR::eHostOrDevice;

	vk::MemoryRequirements2 memoryRequirements;
	m_Device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
															m_Device.getLoader());

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eBuildScratch;
	m_Device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
															m_Device.getLoader());
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeKHR::eUpdateScratch;
	m_Device->getAccelerationStructureMemoryRequirementsKHR(&memoryRequirementsInfo, &memoryRequirements,
															m_Device.getLoader());
	m_ScratchSize = std::max(m_ScratchSize, memoryRequirements.memoryRequirements.size);

	// Create result memory
	m_Memory.allocate(m_ResultSize, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eRayTracingKHR);
	// bind the acceleration structure descriptor to the actual memory that will contain it
	vk::BindAccelerationStructureMemoryInfoKHR bindInfo{};
	bindInfo.accelerationStructure = m_Structure;
	bindInfo.memory = m_Memory.getDeviceMemory();
	bindInfo.memoryOffset = 0;
	bindInfo.deviceIndexCount = 0;
	bindInfo.pDeviceIndices = nullptr;
	m_Device->bindAccelerationStructureMemoryKHR({bindInfo}, m_Device.getLoader());
}

TopLevelAS::~TopLevelAS() { cleanup(); }

void TopLevelAS::cleanup()
{
	if (m_Structure)
		m_Device.getLoader().vkDestroyAccelerationStructureKHR(m_Device, m_Structure, nullptr);
	m_Structure = nullptr;
	m_Memory.cleanup();
	m_InstanceBuffer.cleanup();
}

vk::WriteDescriptorSetAccelerationStructureKHR TopLevelAS::getDescriptorBufferInfo() const
{
	return vk::WriteDescriptorSetAccelerationStructureKHR(1, &m_Structure);
}

void TopLevelAS::Build(bool update, VmaBuffer<uint8_t> scratchBuffer)
{
	if (m_InstanceCnt == 0)
		return;

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

	// build the acceleration structure and store it in the result memory

	auto commandBuffer = m_Device.createOneTimeCmdBuffer();
	commandBuffer->buildAccelerationStructureKHR({buildInfo}, {&m_Offset}, m_Device.getLoader());

	// Ensure that the build will be finished before using the AS using a barrier
	vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteKHR |
										   vk::AccessFlagBits::eAccelerationStructureReadKHR,
									   vk::AccessFlagBits::eAccelerationStructureReadKHR};
	commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildKHR,
								   vk::PipelineStageFlagBits::eRayTracingShaderKHR, {}, 1, &memoryBarrier, 0, nullptr,
								   0, nullptr);

	const auto computeQueue = m_Device.getComputeQueue();
	commandBuffer.submit(computeQueue, true);
}

void TopLevelAS::updateInstances(const std::vector<vk::AccelerationStructureInstanceKHR> &instances)
{
	assert(instances.size() <= m_InstanceCnt);
	m_InstanceBuffer.copyToDevice(instances.data(), instances.size() * sizeof(GeometryInstance));
}

void TopLevelAS::build(const VmaBuffer<uint8_t> &scratchBuffer) { Build(false, scratchBuffer); }

void TopLevelAS::rebuild(const VmaBuffer<uint8_t> &scratchBuffer) { Build(true, scratchBuffer); }

vk::DeviceAddress TopLevelAS::getHandle()
{
	vk::AccelerationStructureDeviceAddressInfoKHR info = {m_Structure};
	return m_Device->getAccelerationStructureAddressKHR(info, m_Device.getLoader());
}

uint32_t TopLevelAS::get_instance_count() const { return m_InstanceCnt; }
