//
// Created by meir on 10/25/19.
//

#ifndef RENDERINGFW_VULKANRTX_SRC_ACCELERATIONSTRUCTURE_H
#define RENDERINGFW_VULKANRTX_SRC_ACCELERATIONSTRUCTURE_H

#include <array>

#include <vulkan/vulkan.hpp>

#include <MathIncludes.h>

#include "VulkanDevice.h"
#include "Buffer.h"

namespace vkrtx
{

struct GeometryInstance
{
	GeometryInstance() = default;
	std::array<float, 12> transform;	  // Transform matrix, containing only the top 3 rows
	uint32_t instanceId : 24;			  // Instance index
	uint32_t mask : 8;					  // Visibility mask
	uint32_t instanceOffset : 24;		  // Index of the hit group which will be invoked when a ray hits the instance
	uint32_t flags : 8;					  // Instance flags, such as culling
	uint64_t accelerationStructureHandle; // Opaque handle of the bottom-level acceleration structure
};

enum AccelerationStructureType // See: https://devblogs.nvidia.com/rtx-best-practices/ for more info
{
	FastestBuild = 0, // Used for geometry like particles
	FastRebuild = 1,  // Low level of detail objects unlikely to be hit, but need to updated frequently
	FastestTrace = 2, // Best for static geometry, provides fastest trace possible
	FastTrace = 3, // Good compromise between fast tracing and build times, best for geometry like player character etc.
};

class TopLevelAS
{
  public:
	explicit TopLevelAS(const VulkanDevice &device, AccelerationStructureType type = FastestTrace,
						uint32_t instanceCount = 32);
	~TopLevelAS();

	void cleanup();

	void updateInstances(const std::vector<GeometryInstance> &instances);
	void build();
	void rebuild();

	uint64_t getHandle();
	[[nodiscard]] uint32_t getInstanceCount() const;

	[[nodiscard]] bool canUpdate() const
	{
		return uint(m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate) > 0;
	}
	[[nodiscard]] vk::AccelerationStructureNV getAccelerationStructure() const { return m_Structure; }
	[[nodiscard]] vk::WriteDescriptorSetAccelerationStructureNV getDescriptorBufferInfo() const;
	Buffer<GeometryInstance> &getInstanceBuffer() { return m_InstanceBuffer; }

  private:
	void Build(bool update);

	// Converts desired type to vulkan build flags
	static vk::BuildAccelerationStructureFlagsNV typeToFlags(AccelerationStructureType type)
	{
		// Different version than bottom level AS, top level acceleration structure do not allow for compaction
		switch (type)
		{
		case (FastestBuild):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild;
		case (FastRebuild):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		case (FastestTrace):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace;
		case (FastTrace):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		default:
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace;
		}
	}

	VulkanDevice m_Device;
	uint32_t m_InstanceCnt = 0;
	vk::DeviceSize m_ResultSize{}, m_ScratchSize{};
	AccelerationStructureType m_Type{};
	vk::BuildAccelerationStructureFlagsNV m_Flags{};
	vk::AccelerationStructureNV m_Structure{};
	Buffer<uint8_t> m_Memory;
	Buffer<GeometryInstance> m_InstanceBuffer;
};

class BottomLevelAS
{
  public:
	BottomLevelAS(VulkanDevice device, const glm::vec4 *vertices, uint32_t vertexCount, const glm::uvec3 *indices,
				  uint32_t indexCount, AccelerationStructureType type);
	~BottomLevelAS();

	void cleanup();
	void updateVertices(const glm::vec4 *vertices, uint32_t vertexCount);

	void build();
	void rebuild();

	uint64_t getHandle();
	[[nodiscard]] uint32_t getVertexCount() const;

	[[nodiscard]] bool canUpdate() const
	{
		return uint(m_Flags & vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate) > 0;
	}
	[[nodiscard]] vk::AccelerationStructureNV getAccelerationStructure() const { return m_Structure; }

  private:
	void build(bool update);

	// Converts desired type to Vulkan build flags
	static vk::BuildAccelerationStructureFlagsNV typeToFlags(AccelerationStructureType type)
	{
		switch (type)
		{
		case (FastestBuild):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild;
		case (FastRebuild):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastBuild |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		case (FastestTrace):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction;
		case (FastTrace):
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowUpdate;
		default:
			return vk::BuildAccelerationStructureFlagBitsNV::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsNV::eAllowCompaction;
		}
	}

	VulkanDevice m_Device;
	vk::DeviceSize m_ResultSize, m_ScratchSize;
	AccelerationStructureType m_Type;
	vk::BuildAccelerationStructureFlagsNV m_Flags;
	vk::GeometryNV m_Geometry;
	vk::AccelerationStructureNV m_Structure;

	Buffer<uint8_t> m_Memory;
	VmaBuffer<glm::vec4> m_Vertices;
	VmaBuffer<glm::uvec3> m_Indices;
};
} // namespace vkrtx

#endif // RENDERINGFW_VULKANRTX_SRC_ACCELERATIONSTRUCTURE_H
