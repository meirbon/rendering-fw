#include "Context.h"

using namespace vkrtx;

ComputePipeline::ComputePipeline(const VulkanDevice &device, const Shader &computeShader) : m_Device(device)
{
	m_ShaderStage = computeShader.getShaderStage(vk::ShaderStageFlagBits::eCompute);
}

ComputePipeline::~ComputePipeline() { cleanup(); }

void ComputePipeline::addPushConstant(vk::PushConstantRange pushConstant)
{
	assert(!m_Generated);
	m_PushConstants.emplace_back(pushConstant);
}

void ComputePipeline::addDescriptorSet(const DescriptorSet *set)
{
	assert(!m_Generated);
	m_DescriptorSets.emplace_back(set);
}

void ComputePipeline::recordPushConstant(vk::CommandBuffer cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data)
{
	assert(m_Generated);
	assert(m_PushConstants.size() > idx);
	assert(m_PushConstants.at(idx).size <= sizeInBytes);

	const auto &pushConstant = m_PushConstants.at(idx);
	cmdBuffer.pushConstants(m_Layout, pushConstant.stageFlags, 0, sizeInBytes, data);
}

void ComputePipeline::recordDispatchCommand(vk::CommandBuffer cmdBuffer, uint32_t width, uint32_t height,
											uint32_t depth)
{
	assert(m_Generated);
	
	// Setup pipeline
	cmdBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_Pipeline);
	if (!m_DescriptorSets.empty())
		cmdBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_Layout, 0, (uint32_t)m_DescriptorSets.size(),
									 m_VkDescriptorSets.data(), 0, nullptr);
	// Dispatch
	cmdBuffer.dispatch(width, height, depth);
}

void ComputePipeline::finalize()
{
	assert(!m_Generated);
	assert(!m_DescriptorSets.empty());

	std::vector<vk::DescriptorSetLayout> layouts;
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

	vk::PipelineLayoutCreateInfo computeLayoutCreateInfo;
	computeLayoutCreateInfo.pNext = nullptr;
	computeLayoutCreateInfo.flags = vk::PipelineLayoutCreateFlags();
	computeLayoutCreateInfo.setLayoutCount = uint32_t(layouts.size());
	computeLayoutCreateInfo.pSetLayouts = layouts.empty() ? nullptr : layouts.data();

	if (m_PushConstants.empty())
	{
		computeLayoutCreateInfo.pushConstantRangeCount = 0;
		computeLayoutCreateInfo.pPushConstantRanges = nullptr;
	}
	else
	{
		computeLayoutCreateInfo.pushConstantRangeCount = uint32_t(m_PushConstants.size());
		computeLayoutCreateInfo.pPushConstantRanges = m_PushConstants.data();
	}

	m_Layout = m_Device->createPipelineLayout(computeLayoutCreateInfo);

	const auto computeCreateInfo =
		vk::ComputePipelineCreateInfo(vk::PipelineCreateFlags(), m_ShaderStage, m_Layout, nullptr);
	m_Pipeline = m_Device->createComputePipeline(nullptr, computeCreateInfo);

	m_Generated = true;
}

void ComputePipeline::cleanup()
{
	m_DescriptorSets.clear();
	m_VkDescriptorSets.clear();
	m_PushConstants.clear();

	if (m_Pipeline)
		m_Device->destroyPipeline(m_Pipeline);
	if (m_Layout)
		m_Device->destroyPipelineLayout(m_Layout);

	m_Pipeline = nullptr;
	m_Layout = nullptr;

	m_Generated = false;
}
