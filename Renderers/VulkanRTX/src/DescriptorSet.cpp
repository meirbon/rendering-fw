//
// Created by meir on 10/25/19.
//

#include <utils.h>
#include "DescriptorSet.h"
using namespace vkrtx;

DescriptorSet::DescriptorSet(const VulkanDevice &device)
	: m_Device(device), m_Buffers(this), m_Images(this), m_AccelerationStructures(this)
{
}

void DescriptorSet::cleanup()
{
	if (m_DescriptorSet)
		m_Device->destroyDescriptorSetLayout(m_Layout);
	if (m_Pool)
		m_Device->destroyDescriptorPool(m_Pool);

	m_DescriptorSet = nullptr;
	m_Layout = nullptr;
	m_Pool = nullptr;

	m_Generated = false;
	m_Dirty = true;
	m_Bindings.clear();
	m_Buffers.clear();
	m_Images.clear();
	m_AccelerationStructures.clear();
}

void DescriptorSet::addBinding(uint32_t binding, uint32_t descriptorCount, vk::DescriptorType type,
							   vk::ShaderStageFlags stage, vk::Sampler *sampler)
{
	if (m_Generated)
		FAILURE("Cannot add bindings after descriptor set has been generated.");
	vk::DescriptorSetLayoutBinding b{};
	b.setBinding(binding);
	b.setDescriptorCount(descriptorCount);
	b.setDescriptorType(type);
	b.setPImmutableSamplers(sampler);
	b.setStageFlags(stage);

	if (m_Bindings.find(binding) != m_Bindings.end())
		FAILURE("Binding collision at %i", binding);

	m_Bindings[binding] = b;
}

void DescriptorSet::finalize()
{
	assert(!m_Generated);
	generatePool();
	generateLayout();
	generateSet();
}

void DescriptorSet::clearBindings()
{
	assert(m_Generated);
	m_Buffers.clear();
	m_Images.clear();
	m_AccelerationStructures.clear();
}

void DescriptorSet::updateSetContents()
{
	assert(m_Generated);
	if (!m_Dirty)
		return;

	// For each resource type, set the actual pointers in the vk::WriteDescriptorSet structures, and
	// write the resulting structures into the descriptor set
	if (!m_Buffers.writeDescs.empty())
	{
		m_Buffers.setPointers();
		m_Device->updateDescriptorSets(static_cast<uint32_t>(m_Buffers.writeDescs.size()), m_Buffers.writeDescs.data(),
									   0, nullptr);
	}

	if (!m_Images.writeDescs.empty())
	{
		m_Images.setPointers();
		m_Device->updateDescriptorSets(static_cast<uint32_t>(m_Images.writeDescs.size()), m_Images.writeDescs.data(), 0,
									   nullptr);
	}

	if (!m_AccelerationStructures.writeDescs.empty())
	{
		m_AccelerationStructures.setPointers();
		m_Device->updateDescriptorSets(static_cast<uint32_t>(m_AccelerationStructures.writeDescs.size()),
									   m_AccelerationStructures.writeDescs.data(), 0, nullptr);
	}
}

void DescriptorSet::bind(uint32_t binding, const std::vector<vk::DescriptorBufferInfo> &bufferInfo)
{
	assert(m_Generated);
	m_Dirty = true;
	m_Buffers.bind(binding, m_Bindings[binding].descriptorType, bufferInfo);
}

void DescriptorSet::bind(uint32_t binding, const std::vector<vk::DescriptorImageInfo> &imageInfo)
{
	assert(m_Generated);
	m_Dirty = true;
	m_Images.bind(binding, m_Bindings[binding].descriptorType, imageInfo);
}

void DescriptorSet::bind(uint32_t binding, const std::vector<vk::WriteDescriptorSetAccelerationStructureNV> &accelInfo)
{
	assert(m_Generated);
	m_Dirty = true;
	m_AccelerationStructures.bind(binding, m_Bindings[binding].descriptorType, accelInfo);
}

void DescriptorSet::generatePool()
{
	m_Generated = true;
	std::vector<vk::DescriptorPoolSize> counters;
	counters.reserve(m_Bindings.size());

	for (const auto &b : m_Bindings)
		counters.emplace_back(b.second.descriptorType, b.second.descriptorCount);

	vk::DescriptorPoolCreateInfo poolInfo{};
	poolInfo.setPoolSizeCount(static_cast<uint32_t>(counters.size()));
	poolInfo.setPPoolSizes(counters.data());
	poolInfo.setMaxSets(1);

	m_Pool = m_Device->createDescriptorPool(poolInfo);
}

void DescriptorSet::generateLayout()
{
	m_Generated = true;
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	bindings.reserve(m_Bindings.size());

	for (const auto &b : m_Bindings)
		bindings.push_back(b.second);

	vk::DescriptorSetLayoutCreateInfo layoutInfo{};
	layoutInfo.setBindingCount((uint32_t)m_Bindings.size());
	layoutInfo.setPBindings(bindings.data());

	m_Layout = m_Device->createDescriptorSetLayout(layoutInfo, nullptr, m_Device.getLoader());
}

void DescriptorSet::generateSet()
{
	m_Generated = true;
	vk::DescriptorSetLayout layouts[] = {m_Layout};
	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.setPNext(nullptr);
	allocInfo.setDescriptorPool(m_Pool);
	allocInfo.setDescriptorSetCount(1);
	allocInfo.setPSetLayouts(layouts);

	m_Device->allocateDescriptorSets(&allocInfo, &m_DescriptorSet, m_Device.getLoader());
}
