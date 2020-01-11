//
// Created by meir on 10/25/19.
//

#ifndef RENDERINGFW_VULKANRTX_SRC_DESCRIPTORSET_H
#define RENDERINGFW_VULKANRTX_SRC_DESCRIPTORSET_H

#include <vulkan/vulkan.hpp>
#include <map>
#include <unordered_map>

#include "VulkanDevice.h"

namespace vkrtx
{
class DescriptorSet
{

  public:
	explicit DescriptorSet(const VulkanDevice &device);
	~DescriptorSet() { cleanup(); }

	void cleanup();

	void addBinding(uint32_t binding, uint32_t descriptorCount, vk::DescriptorType type, vk::ShaderStageFlags stage,
					vk::Sampler *sampler = nullptr);

	void finalize();

	template <typename T, /* Type of the descriptor info, such as vk::DescriptorBufferInfo*/ uint32_t
				  offset /* Offset in the vk::WriteDescriptorSet structure */>
	struct WriteInfo
	{
		WriteInfo(DescriptorSet *set) : descriptorSet(set) {}

		DescriptorSet *descriptorSet;

		std::map<uint32_t, uint32_t> bindingIndices;
		std::vector<vk::WriteDescriptorSet> writeDescs;
		std::vector<std::vector<T>> descContents;

		~WriteInfo() { clear(); }

		void clear()
		{
			bindingIndices.clear();
			writeDescs.clear();
			for (auto &v : descContents)
				v.clear();
			descContents.clear();
		}

		void setPointers()
		{
			for (size_t i = 0; i < writeDescs.size(); i++)
			{
				T **dest = reinterpret_cast<T **>(reinterpret_cast<uint8_t *>(&writeDescs[i]) + offset);
				*dest = descContents[i].data();
			}
		}

		void bind(uint32_t binding, vk::DescriptorType type, const std::vector<T> &info)
		{
			// Initialize the descriptor write, keeping all the resource pointers to NULL since they will
			// be set by setPointers once all resources have been bound
			vk::WriteDescriptorSet descriptorWrite = {};
			descriptorWrite.dstSet = *descriptorSet;
			descriptorWrite.dstBinding = binding;
			descriptorWrite.dstArrayElement = 0;
			descriptorWrite.descriptorType = type;
			descriptorWrite.descriptorCount = static_cast<uint32_t>(info.size());
			descriptorWrite.pBufferInfo = nullptr;
			descriptorWrite.pImageInfo = nullptr;
			descriptorWrite.pTexelBufferView = nullptr;
			descriptorWrite.pNext = nullptr;

			if (bindingIndices.find(binding) != bindingIndices.end()) // Binding already had a value, replace it
			{
				const uint32_t index = bindingIndices[binding];
				writeDescs[index] = descriptorWrite;
				descContents[index] = info;
			}
			else // Add the write descriptor and resource info for later actual binding
			{
				bindingIndices[binding] = static_cast<uint32_t>(writeDescs.size());
				writeDescs.push_back(descriptorWrite);
				descContents.push_back(info);
			}
		}
	};

	// bind a buffer
	void bind(uint32_t binding, const std::vector<vk::DescriptorBufferInfo> &bufferInfo);
	// bind an image
	void bind(uint32_t binding, const std::vector<vk::DescriptorImageInfo> &imageInfo);
	// bind an acceleration structure
	void bind(uint32_t binding, const std::vector<vk::WriteDescriptorSetAccelerationStructureNV> &accelInfo);

	// clear currently bound objects of descriptor set
	void clearBindings();

	// Actually write the binding info into the descriptor set
	void updateSetContents();

	bool isDirty() const { return m_Dirty; }

	operator vk::DescriptorSet() const { return m_DescriptorSet; }
	operator vk::DescriptorSetLayout() const { return m_Layout; }
	operator vk::DescriptorPool() const { return m_Pool; }

	operator vk::DescriptorSet *() { return &m_DescriptorSet; }
	operator vk::DescriptorSetLayout *() { return &m_Layout; }
	operator vk::DescriptorPool *() { return &m_Pool; }

	operator const vk::DescriptorSet *() const { return &m_DescriptorSet; }
	operator const vk::DescriptorSetLayout *() const { return &m_Layout; }
	operator const vk::DescriptorPool *() const { return &m_Pool; }

	vk::DescriptorSet getSet() const { return m_DescriptorSet; }
	vk::DescriptorSetLayout getLayout() const { return m_Layout; }
	vk::DescriptorPool getPool() const { return m_Pool; }

  private:
	void generatePool();
	void generateLayout();
	void generateSet();

	VulkanDevice m_Device;
	vk::DescriptorSet m_DescriptorSet;
	vk::DescriptorSetLayout m_Layout;
	vk::DescriptorPool m_Pool;

	bool m_Dirty = false;
	bool m_Generated = false;

	std::unordered_map<uint32_t, vk::DescriptorSetLayoutBinding> m_Bindings;
	WriteInfo<vk::DescriptorBufferInfo, offsetof(VkWriteDescriptorSet, pBufferInfo)> m_Buffers;
	WriteInfo<vk::DescriptorImageInfo, offsetof(VkWriteDescriptorSet, pImageInfo)> m_Images;
	WriteInfo<vk::WriteDescriptorSetAccelerationStructureNV, offsetof(VkWriteDescriptorSet, pNext)>
		m_AccelerationStructures;
};
} // namespace vkrtx

#endif // RENDERINGFW_VULKANRTX_SRC_DESCRIPTORSET_H
