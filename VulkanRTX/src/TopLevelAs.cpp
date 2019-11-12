//
// Created by meir on 10/25/19.
//

#include "AccelerationStructure.h"

#include "CheckVK.h"

using namespace vkrtx;

TopLevelAS::TopLevelAS(const VulkanDevice &dev, AccelerationStructureType type, uint32_t instanceCount)
	: m_Device(dev), m_InstanceCnt(instanceCount), m_Type(type), m_Flags(typeToFlags(type)), m_Memory(dev),
	  m_InstanceBuffer(dev)
{
	m_InstanceBuffer.allocate(instanceCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
							  vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst);

	vk::AccelerationStructureInfoNV accelerationStructureInfo{};
	accelerationStructureInfo.pNext = nullptr;
	accelerationStructureInfo.type = vk::AccelerationStructureTypeNV::eTopLevel;
	accelerationStructureInfo.flags = m_Flags;
	accelerationStructureInfo.instanceCount = instanceCount;
	accelerationStructureInfo.geometryCount = 0;
	accelerationStructureInfo.pGeometries = nullptr;

	vk::AccelerationStructureCreateInfoNV accelerationStructureCreateInfo{};
	accelerationStructureCreateInfo.pNext = nullptr;
	accelerationStructureCreateInfo.info = accelerationStructureInfo;
	accelerationStructureCreateInfo.compactedSize = 0;

	CheckVK(m_Device.getVkDevice().createAccelerationStructureNV(&accelerationStructureCreateInfo, nullptr,
																 &m_Structure, m_Device.getLoader()));

	// Create a descriptor for the memory requirements, and provide the acceleration structure descriptor
	vk::AccelerationStructureMemoryRequirementsInfoNV memoryRequirementsInfo;
	memoryRequirementsInfo.pNext = nullptr;
	memoryRequirementsInfo.accelerationStructure = m_Structure;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eObject;

	vk::MemoryRequirements2 memoryRequirements;
	m_Device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														   m_Device.getLoader());

	// Size of the resulting AS
	m_ResultSize = memoryRequirements.memoryRequirements.size;

	// Get the largest scratch size requirement
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eBuildScratch;
	m_Device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														   m_Device.getLoader());
	m_ScratchSize = memoryRequirements.memoryRequirements.size;
	memoryRequirementsInfo.type = vk::AccelerationStructureMemoryRequirementsTypeNV::eUpdateScratch;
	m_Device->getAccelerationStructureMemoryRequirementsNV(&memoryRequirementsInfo, &memoryRequirements,
														   m_Device.getLoader());
	m_ScratchSize = std::max(m_ScratchSize, memoryRequirements.memoryRequirements.size);

	// Create result memory
	m_Memory.allocate(m_ResultSize,
					  vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostVisible,
					  vk::BufferUsageFlagBits::eRayTracingNV);
	// bind the acceleration structure descriptor to the actual memory that will contain it
	vk::BindAccelerationStructureMemoryInfoNV bindInfo;
	bindInfo.pNext = nullptr;
	bindInfo.accelerationStructure = m_Structure;
	bindInfo.memory = m_Memory.getDeviceMemory();
	bindInfo.memoryOffset = 0;
	bindInfo.deviceIndexCount = 0;
	bindInfo.pDeviceIndices = nullptr;
	CheckVK(m_Device->bindAccelerationStructureMemoryNV(1, &bindInfo, m_Device.getLoader()));
}

TopLevelAS::~TopLevelAS() { cleanup(); }

void TopLevelAS::cleanup()
{
	if (m_Structure)
		m_Device->destroyAccelerationStructureNV(m_Structure, nullptr, m_Device.getLoader());
	m_Structure = nullptr;
	m_Memory.cleanup();
	m_InstanceBuffer.cleanup();
}

vk::WriteDescriptorSetAccelerationStructureNV TopLevelAS::getDescriptorBufferInfo() const
{
	return vk::WriteDescriptorSetAccelerationStructureNV(1, &m_Structure);
}

void TopLevelAS::Build(bool update)
{
	if (m_InstanceCnt == 0)
		return;

	// build the acceleration structure and store it in the result memory
	vk::AccelerationStructureInfoNV buildInfo = {vk::AccelerationStructureTypeNV::eTopLevel, m_Flags, m_InstanceCnt, 0,
												 nullptr};
	auto scratchBuffer = VmaBuffer<uint8_t>(m_Device, m_ScratchSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
											vk::BufferUsageFlagBits::eRayTracingNV, VMA_MEMORY_USAGE_GPU_ONLY);

	auto commandBuffer = m_Device.createOneTimeCmdBuffer();
	commandBuffer->buildAccelerationStructureNV(&buildInfo, m_InstanceBuffer, 0, update, m_Structure,
												update ? m_Structure : nullptr, scratchBuffer, 0, m_Device.getLoader());

	// Ensure that the build will be finished before using the AS using a barrier
	vk::MemoryBarrier memoryBarrier = {vk::AccessFlagBits::eAccelerationStructureWriteNV |
										   vk::AccessFlagBits::eAccelerationStructureReadNV,
									   vk::AccessFlagBits::eAccelerationStructureReadNV};
	commandBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAccelerationStructureBuildNV,
								   vk::PipelineStageFlagBits::eRayTracingShaderNV, vk::DependencyFlags(), 1,
								   &memoryBarrier, 0, nullptr, 0, nullptr);

	auto computeQueue = m_Device.getComputeQueue();
	commandBuffer.submit(computeQueue, true);
}

void TopLevelAS::updateInstances(const std::vector<GeometryInstance> &instances)
{
	assert(instances.size() <= m_InstanceCnt);
	m_InstanceBuffer.copyToDevice(instances.data(), instances.size() * sizeof(GeometryInstance));
}

void TopLevelAS::build() { Build(false); }

void TopLevelAS::rebuild() { Build(true); }

uint64_t TopLevelAS::getHandle()
{
	uint64_t handle = 0;
	m_Device->getAccelerationStructureHandleNV(m_Structure, sizeof(uint64_t), &handle, m_Device.getLoader());
	assert(handle);
	return handle;
}

uint32_t TopLevelAS::getInstanceCount() const { return m_InstanceCnt; }
