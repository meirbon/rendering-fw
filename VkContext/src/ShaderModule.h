//
// Created by Mèir Noordermeer on 12-09-19.
//

#pragma once

#include <vulkan/vulkan.hpp>

#include <string>

#include "Device.h"

namespace vkc
{

class Device;
class ShaderModule
{
  public:
	ShaderModule() = default;
	ShaderModule(Device &device, const std::string_view &path);
	~ShaderModule();

	static vk::ShaderModule loadModule(Device &device, const std::string_view &path);

	[[nodiscard]] vk::ShaderModule getModule() const { return m_Module; }
	vk::PipelineShaderStageCreateInfo getShaderStage(vk::ShaderStageFlagBits stageFlags);

	operator vk::ShaderModule() const { return m_Module; }

  private:
	Device m_Device;
	vk::ShaderModule m_Module;
};
} // namespace vkc