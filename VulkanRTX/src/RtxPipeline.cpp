//
// Created by meir on 10/25/19.
//

#include "RtxPipeline.h"
#include "CheckVK.h"

#include <utility>

using namespace vkrtx;

void RTXPipeline::ShaderBindingTableGenerator::addRayGenerationProgram(uint32_t groupIdx,
																	   const std::vector<unsigned char> &inlineData)
{
	m_RayGen.emplace_back(SBTEntry(groupIdx, inlineData));
}

void RTXPipeline::ShaderBindingTableGenerator::addMissProgram(uint32_t groupIdx,
															  const std::vector<unsigned char> &inlineData)
{
	m_Miss.emplace_back(groupIdx, inlineData);
}

void RTXPipeline::ShaderBindingTableGenerator::addHitGroup(uint32_t groupIdx,
														   const std::vector<unsigned char> &inlineData)
{
	m_HitGroup.emplace_back(groupIdx, inlineData);
}

vk::DeviceSize
RTXPipeline::ShaderBindingTableGenerator::computeSBTSize(const vk::PhysicalDeviceRayTracingPropertiesNV &props)
{
	// Size of a program identifier
	m_ProgIdSize = props.shaderGroupHandleSize;

	// Compute the entry size of each program type depending on the maximum number of parameters in each category
	m_RayGenEntrySize = getEntrySize(m_RayGen);
	m_MissEntrySize = getEntrySize(m_Miss);
	m_HitGroupEntrySize = getEntrySize(m_HitGroup);

	m_SBTSize = m_RayGenEntrySize * (vk::DeviceSize)m_RayGen.size() + m_MissEntrySize * (vk::DeviceSize)m_Miss.size() +
				m_HitGroupEntrySize * (vk::DeviceSize)m_HitGroup.size();
	return m_SBTSize;
}

void RTXPipeline::ShaderBindingTableGenerator::generate(VulkanDevice &device, vk::Pipeline rtPipeline,
														Buffer<uint8_t> *sbtBuffer)
{
	uint32_t groupCount = static_cast<uint32_t>(m_RayGen.size()) + static_cast<uint32_t>(m_Miss.size()) +
						  static_cast<uint32_t>(m_HitGroup.size());

	// Fetch all the shader handles used in the pipeline, so that they can be written in the SBT
	auto shaderHandleStorage = std::vector<uint8_t>(groupCount * m_ProgIdSize);
	CheckVK(device->getRayTracingShaderGroupHandlesNV(rtPipeline, 0, groupCount, m_ProgIdSize * groupCount,
													  shaderHandleStorage.data(), device.getLoader()));
	std::vector<uint8_t> tempBuffer(m_SBTSize);
	auto data = tempBuffer.data();

	vk::DeviceSize offset = 0;

	// Copy ray generation SBT data
	offset = copyShaderData(rtPipeline, data, m_RayGen, m_RayGenEntrySize, shaderHandleStorage.data());
	data += offset;
	// Copy ray miss SBT data
	offset = copyShaderData(rtPipeline, data, m_Miss, m_MissEntrySize, shaderHandleStorage.data());
	data += offset;
	// Copy ray hit-groups SBT data
	offset = copyShaderData(rtPipeline, data, m_HitGroup, m_HitGroupEntrySize, shaderHandleStorage.data());

	// unmap the SBT
	sbtBuffer->copyToDevice(tempBuffer.data(), m_SBTSize);
}

void RTXPipeline::ShaderBindingTableGenerator::reset()
{
	m_RayGen.clear();
	m_Miss.clear();
	m_HitGroup.clear();

	m_RayGenEntrySize = 0;
	m_MissEntrySize = 0;
	m_HitGroupEntrySize = 0;
	m_ProgIdSize = 0;
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getRayGenSectionSize() const
{
	return m_RayGenEntrySize * static_cast<VkDeviceSize>(m_RayGen.size());
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getRayGenEntrySize() const { return m_RayGenEntrySize; }

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getRayGenOffset() const { return 0; }

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getMissSectionSize() const
{
	return m_MissEntrySize * static_cast<vk::DeviceSize>(m_Miss.size());
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getMissEntrySize() { return m_MissEntrySize; }

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getMissOffset() const { return getRayGenSectionSize(); }

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getHitGroupSectionSize() const
{
	return m_HitGroupEntrySize * static_cast<vk::DeviceSize>(m_HitGroup.size());
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getHitGroupEntrySize() const { return m_HitGroupEntrySize; }

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getHitGroupOffset() const
{
	return getRayGenSectionSize() + getMissSectionSize();
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::copyShaderData(vk::Pipeline pipeline, uint8_t *outputData,
																		const std::vector<SBTEntry> &shaders,
																		vk::DeviceSize entrySize,
																		const uint8_t *shaderHandleStorage)
{
	uint8_t *pData = outputData;
	for (const auto &shader : shaders)
	{
		// Copy the shader identifier that was previously obtained with vkGetRayTracingShaderGroupHandlesNV
		memcpy(pData, shaderHandleStorage + shader.m_GroupIdx * m_ProgIdSize, m_ProgIdSize);

		// Copy all its resources pointers or values in bulk
		if (!shader.m_InlineData.empty())
			memcpy(pData + m_ProgIdSize, shader.m_InlineData.data(), shader.m_InlineData.size());

		pData += entrySize;
	}
	// Return the number of bytes actually written to the output buffer
	return static_cast<uint32_t>(shaders.size()) * entrySize;
}

vk::DeviceSize RTXPipeline::ShaderBindingTableGenerator::getEntrySize(const std::vector<SBTEntry> &entries)
{
	// Find the maximum number of parameters used by a single entry
	size_t maxArgs = 0;
	for (const auto &shader : entries)
	{
		maxArgs = std::max(maxArgs, shader.m_InlineData.size());
	}
	// A SBT entry is made of a program ID and a set of 4-byte parameters (offsets or push constants)
	VkDeviceSize entrySize = m_ProgIdSize + static_cast<VkDeviceSize>(maxArgs);

	// The entries of the shader binding table must be 16-bytes-aligned
	entrySize = (((entrySize) + (16u) - 1u) & ~((16u) - 1u));

	return entrySize;
}

RTXPipeline::ShaderBindingTableGenerator::SBTEntry::SBTEntry(uint32_t groupIdx, std::vector<unsigned char> inlineData)
	: m_GroupIdx(groupIdx), m_InlineData(std::move(inlineData))
{
}

RTXPipeline::RTXPipeline(const VulkanDevice &device) : m_Device(device) {}

void RTXPipeline::cleanup()
{
	m_ShaderIndices.clear();
	m_DescriptorSets.clear();
	m_VkDescriptorSets.clear();
	m_ShaderStages.clear();
	m_ShaderGroups.clear();
	m_CurrentGroupIdx = 0;
	m_MaxRecursionDepth = 5;

	if (m_Pipeline)
		m_Device->destroyPipeline(m_Pipeline);
	if (m_Layout)
		m_Device->destroyPipelineLayout(m_Layout);
	if (SBTBuffer)
		delete SBTBuffer;

	m_Pipeline = nullptr;
	m_Layout = nullptr;
	SBTBuffer = nullptr;

	m_SBTGenerator = {};
	m_Generated = false;
}

uint32_t RTXPipeline::addEmptyHitGroup()
{
	assert(!m_Generated);
	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext(nullptr);
	groupInfo.setType(vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup);
	groupInfo.setGeneralShader(VK_SHADER_UNUSED_NV);
	groupInfo.setClosestHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setAnyHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setIntersectionShader(VK_SHADER_UNUSED_NV);

	m_ShaderGroups.push_back(groupInfo);
	const auto idx = m_CurrentGroupIdx;
	m_CurrentGroupIdx++;
	m_ShaderIndices.emplace_back(std::make_pair(HITGROUP, idx));
	return idx;
}

uint32_t RTXPipeline::addHitGroup(const RTXHitGroup &hitGroup)
{
	assert(!m_Generated);
	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext(nullptr);
	groupInfo.setType(vk::RayTracingShaderGroupTypeNV::eTrianglesHitGroup);
	if (hitGroup.generalShader)
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext(nullptr);
		stageCreate.setStage(vk::ShaderStageFlagBits::eCallableNV);
		stageCreate.setModule(*hitGroup.generalShader);
		stageCreate.setPName("main");
		stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
		stageCreate.setPSpecializationInfo(nullptr);

		m_ShaderStages.emplace_back(stageCreate);
		const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);
		groupInfo.setGeneralShader(shaderIdx);
	}
	else
		groupInfo.setGeneralShader(VK_SHADER_UNUSED_NV);
	if (hitGroup.closestHitShader)
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext(nullptr);
		stageCreate.setStage(vk::ShaderStageFlagBits::eClosestHitNV);
		stageCreate.setModule(*hitGroup.closestHitShader);
		stageCreate.setPName("main");
		stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
		stageCreate.setPSpecializationInfo(nullptr);

		m_ShaderStages.emplace_back(stageCreate);
		const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);
		groupInfo.setClosestHitShader(shaderIdx);
	}
	else
		groupInfo.setClosestHitShader(VK_SHADER_UNUSED_NV);
	if (hitGroup.anyHitShader)
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext(nullptr);
		stageCreate.setStage(vk::ShaderStageFlagBits::eAnyHitNV);
		stageCreate.setModule(*hitGroup.anyHitShader);
		stageCreate.setPName("main");
		stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
		stageCreate.setPSpecializationInfo(nullptr);

		m_ShaderStages.emplace_back(stageCreate);
		const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);
		groupInfo.setAnyHitShader(shaderIdx);
	}
	else
		groupInfo.setAnyHitShader(VK_SHADER_UNUSED_NV);
	if (hitGroup.intersectionShader)
	{
		vk::PipelineShaderStageCreateInfo stageCreate{};
		stageCreate.setPNext(nullptr);
		stageCreate.setStage(vk::ShaderStageFlagBits::eIntersectionNV);
		stageCreate.setModule(*hitGroup.intersectionShader);
		stageCreate.setPName("main");
		stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
		stageCreate.setPSpecializationInfo(nullptr);

		m_ShaderStages.emplace_back(stageCreate);
		const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);
		groupInfo.setIntersectionShader(shaderIdx);
	}
	else
		groupInfo.setIntersectionShader(VK_SHADER_UNUSED_NV);

	m_ShaderGroups.push_back(groupInfo);
	const auto idx = m_CurrentGroupIdx;
	m_CurrentGroupIdx++;
	m_ShaderIndices.emplace_back(std::make_pair(HITGROUP, idx));
	return idx;
}

uint32_t RTXPipeline::addRayGenShaderStage(vk::ShaderModule module)
{
	assert(!m_Generated);
	vk::PipelineShaderStageCreateInfo stageCreate{};
	stageCreate.setPNext(nullptr);
	stageCreate.setStage(vk::ShaderStageFlagBits::eRaygenNV);
	stageCreate.setModule(module);
	stageCreate.setPName("main");
	stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
	stageCreate.setPSpecializationInfo(nullptr);

	m_ShaderStages.emplace_back(stageCreate);
	const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);

	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext(nullptr);
	groupInfo.setType(vk::RayTracingShaderGroupTypeNV::eGeneral);
	groupInfo.setGeneralShader(shaderIdx);
	groupInfo.setClosestHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setAnyHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setIntersectionShader(VK_SHADER_UNUSED_NV);
	m_ShaderGroups.emplace_back(groupInfo);

	m_ShaderIndices.emplace_back(std::make_pair(RAYGEN, m_CurrentGroupIdx));

	return m_CurrentGroupIdx++;
}

uint32_t RTXPipeline::addMissShaderStage(vk::ShaderModule module)
{
	vk::PipelineShaderStageCreateInfo stageCreate{};
	stageCreate.setPNext(nullptr);
	stageCreate.setStage(vk::ShaderStageFlagBits::eMissNV);
	stageCreate.setModule(module);
	stageCreate.setPName("main");
	stageCreate.setFlags(vk::PipelineShaderStageCreateFlags());
	stageCreate.setPSpecializationInfo(nullptr);

	m_ShaderStages.emplace_back(stageCreate);
	const auto shaderIdx = static_cast<uint32_t>(m_ShaderStages.size() - 1);

	vk::RayTracingShaderGroupCreateInfoNV groupInfo{};
	groupInfo.setPNext(nullptr);
	groupInfo.setType(vk::RayTracingShaderGroupTypeNV::eGeneral);
	groupInfo.setGeneralShader(shaderIdx);
	groupInfo.setClosestHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setAnyHitShader(VK_SHADER_UNUSED_NV);
	groupInfo.setIntersectionShader(VK_SHADER_UNUSED_NV);
	m_ShaderGroups.emplace_back(groupInfo);

	m_ShaderIndices.emplace_back(std::make_pair(MISS, m_CurrentGroupIdx));

	return m_CurrentGroupIdx++;
}

void RTXPipeline::setMaxRecursionDepth(uint32_t maxDepth) { m_MaxRecursionDepth = maxDepth; }

void RTXPipeline::addPushConstant(vk::PushConstantRange pushConstant) { m_PushConstants.emplace_back(pushConstant); }

void RTXPipeline::addDescriptorSet(const DescriptorSet *set) { m_DescriptorSets.emplace_back(set); }

void RTXPipeline::finalize()
{
	assert(!m_Generated);

	std::vector<vk::DescriptorSetLayout> layouts;
	vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
	if (!m_DescriptorSets.empty())
	{
		layouts.resize(m_DescriptorSets.size());
		m_VkDescriptorSets.resize(m_DescriptorSets.size());
		for (size_t i = 0; i < m_DescriptorSets.size(); i++)
		{
			layouts.at(i) = m_DescriptorSets.at(i)->getLayout();
			m_VkDescriptorSets.at(i) = m_DescriptorSets.at(i)->getSet();
		}
	}

	pipelineLayoutCreateInfo.setPNext(nullptr);
	pipelineLayoutCreateInfo.setFlags(vk::PipelineLayoutCreateFlags());
	pipelineLayoutCreateInfo.setSetLayoutCount(uint32_t(layouts.size()));
	pipelineLayoutCreateInfo.setPSetLayouts(layouts.empty() ? nullptr : layouts.data());

	if (!m_PushConstants.empty())
	{
		pipelineLayoutCreateInfo.setPushConstantRangeCount(uint32_t(m_PushConstants.size()));
		pipelineLayoutCreateInfo.setPPushConstantRanges(m_PushConstants.data());
	}
	else
	{
		pipelineLayoutCreateInfo.setPushConstantRangeCount(0);
		pipelineLayoutCreateInfo.setPPushConstantRanges(nullptr);
	}

	CheckVK(m_Device->createPipelineLayout(&pipelineLayoutCreateInfo, nullptr, &m_Layout));

	vk::RayTracingPipelineCreateInfoNV rayPipelineInfo{};
	rayPipelineInfo.setPNext(nullptr);
	rayPipelineInfo.setFlags(vk::PipelineCreateFlags());
	rayPipelineInfo.setStageCount((uint32_t)m_ShaderStages.size());
	rayPipelineInfo.setPStages(m_ShaderStages.data());
	rayPipelineInfo.setGroupCount((uint32_t)m_ShaderGroups.size());
	rayPipelineInfo.setPGroups(m_ShaderGroups.data());
	rayPipelineInfo.setMaxRecursionDepth(1);
	rayPipelineInfo.setLayout(m_Layout);
	rayPipelineInfo.setBasePipelineHandle(nullptr);
	rayPipelineInfo.setBasePipelineIndex(0);

	CheckVK(m_Device->createRayTracingPipelinesNV(nullptr, 1, &rayPipelineInfo, nullptr, &m_Pipeline,
												  m_Device.getLoader()));

	for (const auto &shader : m_ShaderIndices)
	{
		const auto type = shader.first;
		const auto index = shader.second;

		switch (type)
		{
		case (RAYGEN):
			m_SBTGenerator.addRayGenerationProgram(index, {});
			break;
		case (MISS):
			m_SBTGenerator.addMissProgram(index, {});
			break;
		case (HITGROUP):
			m_SBTGenerator.addHitGroup(index, {});
			break;
		}
	}

	vk::PhysicalDeviceProperties2 props;
	vk::PhysicalDeviceRayTracingPropertiesNV rtProperties;
	props.pNext = &rtProperties;
	m_Device.getPhysicalDevice().getProperties2(&props, m_Device.getLoader());
	const auto sbtSize = m_SBTGenerator.computeSBTSize(rtProperties);
	SBTBuffer = new Buffer<uint8_t>(m_Device, sbtSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
									vk::BufferUsageFlagBits::eRayTracingNV | vk::BufferUsageFlagBits::eTransferDst);
	m_SBTGenerator.generate(m_Device, m_Pipeline, SBTBuffer);

	m_Generated = true;
}

void RTXPipeline::recordPushConstant(vk::CommandBuffer cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data)
{
	assert(m_Generated);
	assert(m_PushConstants.size() > idx);
	assert(m_PushConstants.at(idx).size <= sizeInBytes);

	const auto &pushConstant = m_PushConstants.at(idx);
	cmdBuffer.pushConstants(m_Layout, pushConstant.stageFlags, 0, sizeInBytes, data);
}

void RTXPipeline::recordTraceCommand(vk::CommandBuffer cmdBuffer, uint32_t width, uint32_t height, uint32_t depth)
{
	assert(m_Generated);

	// Setup pipeline
	cmdBuffer.bindPipeline(vk::PipelineBindPoint::eRayTracingNV, m_Pipeline, m_Device.getLoader());
	if (!m_DescriptorSets.empty())
		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingNV, m_Layout, 0,
									 uint32_t(m_VkDescriptorSets.size()), m_VkDescriptorSets.data(), 0, nullptr);

	const vk::Buffer shaderBindingTableBuffer = *SBTBuffer;
	const auto rayGenOffset = m_SBTGenerator.getRayGenOffset();
	const auto missOffset = m_SBTGenerator.getMissOffset();
	const auto hitOffset = m_SBTGenerator.getHitGroupOffset();
	const auto rayGenSize = m_SBTGenerator.getRayGenSectionSize();
	const auto missSize = m_SBTGenerator.getMissSectionSize();
	const auto hitSize = m_SBTGenerator.getHitGroupSectionSize();

	// Intersect rays
	cmdBuffer.traceRaysNV(shaderBindingTableBuffer, rayGenOffset, shaderBindingTableBuffer, missOffset, missSize,
						  shaderBindingTableBuffer, hitOffset, hitSize, nullptr, 0, 0, width, height, depth,
						  m_Device.getLoader());
}
