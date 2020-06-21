#pragma once

namespace vkrtx
{

struct GeometryInstance
{
	GeometryInstance() = default;
	float transform[12];				  // Transform matrix, containing only the top 3 rows
	uint32_t instanceId : 24;			  // Instance index
	uint32_t mask : 8;					  // Visibility mask
	uint32_t instanceOffset : 24;		  // Index of the hit group which will be iKHRoked when a ray hits the instance
	uint32_t flags : 8;					  // Instance flags, such as culling
	uint64_t accelerationStructureHandle; // Opaque handle of the bottom-level acceleration structure
};

enum AccelerationStructureType // See: https://devblogs.KHRidia.com/rtx-best-practices/ for more info
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

	void updateInstances(const std::vector<vk::AccelerationStructureInstanceKHR> &instances);
	void build(const VmaBuffer<uint8_t> &scratchBuffer);
	void rebuild(const VmaBuffer<uint8_t> &scratchBuffer);

	vk::DeviceAddress getHandle();
	[[nodiscard]] uint32_t get_instance_count() const;

	[[nodiscard]] bool canUpdate() const
	{
		return uint(m_Flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate) > 0;
	}
	[[nodiscard]] vk::AccelerationStructureKHR getAccelerationStructure() const { return m_Structure; }
	[[nodiscard]] vk::WriteDescriptorSetAccelerationStructureKHR getDescriptorBufferInfo() const;
	VmaBuffer<vk::AccelerationStructureInstanceKHR> &getInstanceBuffer() { return m_InstanceBuffer; }

  private:
	void Build(bool update, VmaBuffer<uint8_t> scratchBuffer);

	// CoKHRerts desired type to vulkan build flags
	static vk::BuildAccelerationStructureFlagsKHR typeToFlags(AccelerationStructureType type)
	{
		// Different version than bottom level AS, top level acceleration structure do not allow for compaction
		switch (type)
		{
		case (FastestBuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
		case (FastRebuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		case (FastestTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		case (FastTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		default:
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		}
	}

	VulkanDevice m_Device;
	uint32_t m_InstanceCnt = 0;
	vk::DeviceSize m_ResultSize{}, m_ScratchSize{};
	AccelerationStructureType m_Type{};
	vk::BuildAccelerationStructureFlagsKHR m_Flags{};

	vk::AccelerationStructureCreateGeometryTypeInfoKHR m_GeometryTypeInfo{};
	vk::AccelerationStructureGeometryInstancesDataKHR m_GeometryInstances{};
	vk::AccelerationStructureGeometryKHR m_Geometry{};
	vk::AccelerationStructureBuildOffsetInfoKHR m_Offset{};
	vk::AccelerationStructureKHR m_Structure{};

	VmaBuffer<uint8_t> m_Memory;
	VmaBuffer<vk::AccelerationStructureInstanceKHR> m_InstanceBuffer;
};

class BottomLevelAS
{
  public:
	BottomLevelAS(VulkanDevice device, const glm::vec4 *vertices, uint32_t vertexCount, const glm::uvec3 *indices,
				  uint32_t indexCount, AccelerationStructureType type);
	~BottomLevelAS();

	void cleanup();
	void updateVertices(const glm::vec4 *vertices, uint32_t vertexCount);

	void build(const VmaBuffer<uint8_t> &scratchBuffer);
	void rebuild(const VmaBuffer<uint8_t> &scratchBuffer);

	vk::DeviceAddress getHandle();
	[[nodiscard]] uint32_t getVertexCount() const;

	[[nodiscard]] bool canUpdate() const
	{
		return uint(m_Flags & vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate) > 0;
	}
	[[nodiscard]] vk::AccelerationStructureKHR getAccelerationStructure() const { return m_Structure; }

  private:
	void build(bool update, VmaBuffer<uint8_t> scratchBuffer);

	// CoKHRerts desired type to Vulkan build flags
	static vk::BuildAccelerationStructureFlagsKHR typeToFlags(AccelerationStructureType type)
	{
#if 1
		switch (type)
		{
		case (FastestBuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
		case (FastRebuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		case (FastestTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		case (FastTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		default:
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		}
#else
		switch (type)
		{
		case (FastestBuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild;
		case (FastRebuild):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastBuild |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		case (FastestTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
		case (FastTrace):
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowUpdate;
		default:
			return vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace |
				   vk::BuildAccelerationStructureFlagBitsKHR::eAllowCompaction;
		}
#endif
	}

	int m_PrimCount;
	VulkanDevice m_Device;
	vk::DeviceSize m_ResultSize, m_ScratchSize;
	AccelerationStructureType m_Type;
	vk::BuildAccelerationStructureFlagsKHR m_Flags{};
	vk::GeometryFlagsKHR m_GeometryFlags{};
	vk::GeometryTypeKHR m_GeometryType{};
	vk::AccelerationStructureCreateGeometryTypeInfoKHR m_GeometryTypeInfo{};
	vk::AccelerationStructureGeometryTrianglesDataKHR m_GeometryTriangles{};
	vk::AccelerationStructureGeometryKHR m_Geometry{};
	vk::AccelerationStructureBuildOffsetInfoKHR m_Offset{};
	vk::AccelerationStructureKHR m_Structure{};

	Buffer<uint8_t> m_Memory;
	VmaBuffer<glm::vec4> m_Vertices;
	VmaBuffer<glm::uvec3> m_Indices;
};
} // namespace vkrtx
