//
// Created by meir on 10/5/19.
//

#ifndef RENDERINGFW_VKCONTEXT_SRC_RENDERPASS_H
#define RENDERINGFW_VKCONTEXT_SRC_RENDERPASS_H

#include <cassert>
#include <vulkan/vulkan.hpp>

#include "VulkanDevice.h"

namespace vkc
{
struct Subpass
{
	vk::PipelineBindPoint bindPoint = vk::PipelineBindPoint::eGraphics;
	std::vector<vk::AttachmentReference> attachments;
};

class RenderPass
{
  public:
	RenderPass(VulkanDevice device) : m_Device(device) {}
	~RenderPass() { cleanup(); }

	uint32_t addColorAttachment(vk::Format format, vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
								vk::AttachmentStoreOp storeOp = vk::AttachmentStoreOp::eStore,
								vk::AttachmentLoadOp stencilLoadOp = vk::AttachmentLoadOp::eClear,
								vk::AttachmentStoreOp stencilStoreOp = vk::AttachmentStoreOp::eStore,
								vk::SampleCountFlagBits samples = vk::SampleCountFlagBits::e1,
								vk::ImageLayout initialLayout = vk::ImageLayout::eUndefined,
								vk::ImageLayout finalLayout = vk::ImageLayout::ePresentSrcKHR);
	void addSubpass(const Subpass &pass);

	void finalize();

	void cleanup();

	operator vk::RenderPass() const
	{
		assert(m_Finalized);
		return m_RenderPass;
	}

  private:
	bool m_Finalized = false;
	VulkanDevice m_Device;
	vk::RenderPass m_RenderPass;
	std::vector<vk::AttachmentDescription> m_AttachmentDescriptions;
	std::vector<vk::SubpassDescription> m_SubpassDescriptions;
	std::vector<Subpass> m_Subpasses;
};

} // namespace vkc

#endif // RENDERINGFW_VKCONTEXT_SRC_RENDERPASS_H
