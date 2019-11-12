//
// Created by meir on 10/25/19.
//

#include "Shader.h"

#include <utils/File.h>

using namespace vkrtx;

std::string Shader::BaseFolder = "";

inline bool IsSPIR_V(const std::string_view &fileName)
{
	const std::string_view extention = fileName.substr(fileName.size() - 4, 4);
	return extention == ".spv";
}

Shader::Shader(const VulkanDevice &device, const std::string_view &fileName,
			   const std::vector<std::pair<std::string, std::string>> &definitions)
	: m_Device(device)
{
	const auto cwd = rfw::utils::file::get_working_path();
	std::string fileLocation = cwd + "/" + BaseFolder + std::string(fileName.data());

	// Sanity check
	if (!rfw::utils::file::exists(fileLocation))
	{
		std::string spvLocation = fileLocation + ".spv";
		if (!rfw::utils::file::exists(spvLocation))
		{
			char buffer[1024];
			sprintf(buffer, "File: \"%s\" does not exist.", fileLocation.data());
			FAILURE(buffer);
		}
		else
		{
			fileLocation = spvLocation;
		}
	}

	const auto source = readFile(fileLocation);
	vk::ShaderModuleCreateInfo shaderModuleCreateInfo =
		vk::ShaderModuleCreateInfo(vk::ShaderModuleCreateFlags(), source.size(), (uint32_t *)(source.data()));
	m_Module = m_Device->createShaderModule(shaderModuleCreateInfo);
}

Shader::~Shader() { Cleanup(); }

void Shader::Cleanup()
{
	if (m_Module)
	{
		m_Device->destroyShaderModule(m_Module);
		m_Module = nullptr;
	}
}

vk::PipelineShaderStageCreateInfo Shader::getShaderStage(vk::ShaderStageFlagBits stage) const
{
	vk::PipelineShaderStageCreateInfo result{};
	result.setPNext(nullptr);
	result.setStage(stage);
	result.setModule(m_Module);
	result.setPName("main");
	result.setFlags(vk::PipelineShaderStageCreateFlags());
	result.setPSpecializationInfo(nullptr);
	return result;
}

std::vector<char> Shader::readFile(const std::string_view &fileName)
{
	std::ifstream fileStream(fileName.data(), std::ios::binary | std::ios::in | std::ios::ate);
	if (!fileStream.is_open())
		FAILURE("Could not open file.");

	const size_t size = fileStream.tellg();
	fileStream.seekg(0, std::ios::beg);
	std::vector<char> data(size);
	fileStream.read(data.data(), size);
	fileStream.close();
	return data;
}

std::string Shader::readTextFile(const std::string_view &fileName)
{
	std::string buffer;
	std::ifstream fileStream(fileName.data());
	if (!fileStream.is_open())
		FAILURE("Could not open file.");

	std::string temp;
	while (getline(fileStream, temp))
		buffer.append(temp), buffer.append("\n");

	fileStream >> buffer;
	fileStream.close();
	return buffer;
}