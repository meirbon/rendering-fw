//
// Created by Mèir Noordermeer on 12-09-19.
//

#include "ShaderModule.h"
#include "VulkanDevice.h"

#include <utils/File.h>

using namespace vkc;

static std::vector<char> read_file(const std::string_view &filename)
{
	std::ifstream file(filename.data(), std::ios::ate | std::ios::binary);

	if (!file.is_open())
		throw std::runtime_error("Could not open file!");

	size_t fileSize = static_cast<size_t>(file.tellg());
	std::vector<char> buffer(fileSize);

	file.seekg(0);
	file.read(buffer.data(), fileSize);
	file.close();
	return buffer;
}

ShaderModule::ShaderModule(VulkanDevice &device, const std::string_view &path) : m_Device(device)
{
	m_Module = loadModule(device, path);
}

ShaderModule::~ShaderModule() { m_Device->destroyShaderModule(m_Module); }

vk::ShaderModule ShaderModule::loadModule(VulkanDevice &device, const std::string_view &path)
{
	vk::ShaderModuleCreateInfo createInfo{};

	const auto source = read_file(path);

	createInfo.setCodeSize(source.size());
	createInfo.setPCode(reinterpret_cast<const uint32_t *>(source.data()));

	return device->createShaderModule(createInfo, nullptr);
}

vk::PipelineShaderStageCreateInfo ShaderModule::getShaderStage(vk::ShaderStageFlagBits stageFlags)
{
	return vk::PipelineShaderStageCreateInfo({}, stageFlags, m_Module, "main", nullptr);
}
