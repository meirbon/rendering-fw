//
// Created by Mèir Noordermeer on 12-09-19.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include <string>

#include "VulkanDevice.h"

namespace vkc
{

class Device;
class ShaderModule
{
  public:
	ShaderModule() = default;
	ShaderModule(VulkanDevice &device, const std::string_view &path);
	~ShaderModule();

	static vk::ShaderModule loadModule(VulkanDevice &device, const std::string_view &path);

	[[nodiscard]] vk::ShaderModule getModule() const { return m_Module; }
	vk::PipelineShaderStageCreateInfo getShaderStage(vk::ShaderStageFlagBits stageFlags);

	operator vk::ShaderModule() const { return m_Module; }

  private:
	VulkanDevice m_Device;
	vk::ShaderModule m_Module;
};
} // namespace vkc