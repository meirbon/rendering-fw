//
// Created by meir on 10/25/19.
//

#ifndef RENDERINGFW_VULKANRTX_SRC_COMPUTEPIPELINE_H
#define RENDERINGFW_VULKANRTX_SRC_COMPUTEPIPELINE_H

#include <vulkan/vulkan.hpp>
#include "VulkanDevice.h"
#include "Shader.h"
#include "DescriptorSet.h"

namespace vkrtx
{
class ComputePipeline
{
  public:
	ComputePipeline(const VulkanDevice &device, const Shader &computeShader);
	~ComputePipeline();

	void addPushConstant(vk::PushConstantRange pushConstant);
	void addDescriptorSet(const DescriptorSet *set);

	void recordPushConstant(vk::CommandBuffer cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data);
	void recordDispatchCommand(vk::CommandBuffer cmdBuffer, uint32_t width, uint32_t height = 1, uint32_t depth = 1);

	void finalize();

	void cleanup();

  private:
	bool m_Generated = false;
	VulkanDevice m_Device;
	vk::PipelineShaderStageCreateInfo m_ShaderStage;
	std::vector<const DescriptorSet *> m_DescriptorSets;
	std::vector<vk::DescriptorSet> m_VkDescriptorSets;
	std::vector<vk::PushConstantRange> m_PushConstants;
	vk::Pipeline m_Pipeline = nullptr;
	vk::PipelineLayout m_Layout = nullptr;
};
} // namespace vkrtx

#endif // RENDERINGFW_VULKANRTX_SRC_COMPUTEPIPELINE_H
