//
// Created by meir on 10/25/19.
//

#ifndef RENDERINGFW_VULKANRTX_SRC_SHADER_H
#define RENDERINGFW_VULKANRTX_SRC_SHADER_H

#include <vulkan/vulkan.hpp>
#include "VulkanDevice.h"

#include <string>
#include <string_view>
#include <vector>
#include <tuple>

namespace vkrtx
{

class Shader
{
  public:
	Shader() = default;
	Shader(const VulkanDevice &device, const std::string_view &fileName,
		   const std::vector<std::pair<std::string, std::string>> &definitions = {});
	~Shader();

	void Cleanup();
	[[nodiscard]] vk::PipelineShaderStageCreateInfo getShaderStage(vk::ShaderStageFlagBits stage) const;

	operator vk::ShaderModule() const { return m_Module; }

	static std::string BaseFolder;
	static std::string BSDFFolder;

  private:
	VulkanDevice m_Device;
	vk::ShaderModule m_Module = nullptr;

	static std::vector<char> readFile(const std::string_view &fileName);
	static std::string readTextFile(const std::string_view &fileName);
};
} // namespace vkrtx

#endif // RENDERINGFW_VULKANRTX_SRC_SHADER_H
