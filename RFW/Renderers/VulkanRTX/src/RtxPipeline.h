#pragma once
#pragma once

#include <vulkan/vulkan.hpp>
#include "VulkanDevice.h"
#include "Shader.h"
#include "DescriptorSet.h"
#include "Buffer.h"

#include <vector>

namespace vkrtx
{
struct RTXHitGroup
{
	explicit RTXHitGroup(const Shader *general = nullptr, const Shader *closestHit = nullptr,
						 const Shader *anyHit = nullptr, const Shader *intersection = nullptr)
		: generalShader(general), closestHitShader(closestHit), anyHitShader(anyHit), intersectionShader(intersection)
	{
	}

	const Shader *generalShader = nullptr;
	const Shader *closestHitShader = nullptr;
	const Shader *anyHitShader = nullptr;
	const Shader *intersectionShader = nullptr;
};
class RTXPipeline
{
	class ShaderBindingTableGenerator
	{
	  public:
		// Add a ray generation program by name, with its list of data pointers or values according to
		// the layout of its root signature
		void addRayGenerationProgram(uint32_t groupIdx, const std::vector<unsigned char> &inlineData);

		// Add a miss program by name, with its list of data pointers or values according to
		// the layout of its root signature
		void addMissProgram(uint32_t groupIdx, const std::vector<unsigned char> &inlineData);

		// Add a hit group by name, with its list of data pointers or values according to
		// the layout of its root signature
		void addHitGroup(uint32_t groupIdx, const std::vector<unsigned char> &inlineData);

		/// Compute the size of the SBT based on the set of programs and hit groups it contains
		vk::DeviceSize computeSBTSize(const vk::PhysicalDeviceRayTracingPropertiesKHR &props);

		// build the SBT and store it into sbtBuffer, which has to be preallocated on the upload heap.
		// Access to the ray tracing pipeline object is required to fetch program identifiers using their names
		void generate(VulkanDevice &device, vk::Pipeline rtPipeline, Buffer<uint8_t> &sbtBuffer);

		void reset(); /// reset the sets of programs and hit groups

		[[nodiscard]] vk::DeviceSize getRayGenSectionSize() const;
		// Get the size in bytes of one ray generation program entry in the SBT
		[[nodiscard]] vk::DeviceSize getRayGenEntrySize() const;

		[[nodiscard]] vk::DeviceSize getRayGenOffset() const;

		// Get the size in bytes of the SBT section dedicated to miss programs
		[[nodiscard]] vk::DeviceSize getMissSectionSize() const;
		// Get the size in bytes of one miss program entry in the SBT
		vk::DeviceSize getMissEntrySize();

		[[nodiscard]] vk::DeviceSize getMissOffset() const;

		// Get the size in bytes of the SBT section dedicated to hit groups
		[[nodiscard]] vk::DeviceSize getHitGroupSectionSize() const;
		// Get the size in bytes of hit group entry in the SBT
		[[nodiscard]] vk::DeviceSize getHitGroupEntrySize() const;

		[[nodiscard]] vk::DeviceSize getHitGroupOffset() const;

	  private:
		// Wrapper for SBT entries, each consisting of the name of the program and a list of values,
		// which can be either offsets or raw 32-bit constants
		struct SBTEntry
		{
			SBTEntry(uint32_t groupIdx, std::vector<unsigned char> inlineData);

			uint32_t m_GroupIdx;
			const std::vector<unsigned char> m_InlineData;
		};

		// For each entry, copy the shader identifier followed by its resource pointers and/or root
		// constants in outputData, with a stride in bytes of entrySize, and returns the size in bytes
		// actually written to outputData.
		vk::DeviceSize copyShaderData(vk::Pipeline pipeline, uint8_t *outputData, const std ::vector<SBTEntry> &shaders,
									  vk::DeviceSize entrySize, const uint8_t *shaderHandleStorage);

		// Compute the size of the SBT entries for a set of entries, which is determined by the maximum
		// number of parameters of their root signature
		vk::DeviceSize getEntrySize(const std::vector<SBTEntry> &entries);

		std::vector<SBTEntry> m_RayGen;	  // Ray generation shader entries
		std::vector<SBTEntry> m_Miss;	  // Miss shader entries
		std::vector<SBTEntry> m_HitGroup; /// Hit group entries

		// For each category, the size of an entry in the SBT depends on the maximum number of resources
		// used by the shaders in that category.The helper computes those values automatically in
		// getEntrySize()
		vk::DeviceSize m_RayGenEntrySize = 0;
		vk::DeviceSize m_MissEntrySize = 0;
		vk::DeviceSize m_HitGroupEntrySize = 0;

		// The program names are translated into program identifiers.The size in bytes of an identifier
		// is provided by the device and is the same for all categories.
		vk::DeviceSize m_ProgIdSize = 0;
		vk::DeviceSize m_SBTSize = 0;
	};

  public:
	explicit RTXPipeline(const VulkanDevice &device);
	~RTXPipeline() { cleanup(); }

	void cleanup();
	// Helper function for use-cases like shadow-rays, any shader type requires at least 1 hit group to be usable
	uint32_t addEmptyHitGroup();
	uint32_t addHitGroup(const RTXHitGroup &hitGroup);
	uint32_t addRayGenShaderStage(vk::ShaderModule module);
	uint32_t addMissShaderStage(vk::ShaderModule module);
	void setMaxRecursionDepth(uint32_t maxDepth);
	void addPushConstant(vk::PushConstantRange pushConstant);
	void addDescriptorSet(const DescriptorSet *set);

	void finalize();

	void recordPushConstant(vk::CommandBuffer cmdBuffer, uint32_t idx, uint32_t sizeInBytes, void *data);
	void recordTraceCommand(vk::CommandBuffer cmdBuffer, uint32_t width, uint32_t height = 1, uint32_t depth = 1);

	operator vk::Pipeline() const
	{
		assert(m_Generated);
		return m_Pipeline;
	}
	operator vk::PipelineLayout() const
	{
		assert(m_Generated);
		return m_Layout;
	}

  private:
	enum ShaderType
	{
		RAYGEN = 0,
		MISS = 1,
		HITGROUP = 2
	};

	bool m_Generated = false;
	VulkanDevice m_Device;
	std::vector<std::pair<ShaderType, uint32_t>> m_ShaderIndices;
	std::vector<const DescriptorSet *> m_DescriptorSets;
	std::vector<vk::DescriptorSet> m_VkDescriptorSets;
	std::vector<vk::PipelineShaderStageCreateInfo> m_ShaderStages;
	std::vector<vk::RayTracingShaderGroupCreateInfoKHR> m_ShaderGroups;

	uint32_t m_CurrentGroupIdx = 0;
	uint32_t m_MaxRecursionDepth = 5;

	// Pipeline
	vk::PhysicalDeviceRayTracingPropertiesKHR m_RtProps;
	vk::Pipeline m_Pipeline = nullptr;
	vk::PipelineLayout m_Layout = nullptr;
	std::unique_ptr<Buffer<uint8_t>> m_SBTBuffer = nullptr;

	std::vector<vk::PushConstantRange> m_PushConstants;
	ShaderBindingTableGenerator m_SBTGenerator;
};
} // namespace vkrtx
