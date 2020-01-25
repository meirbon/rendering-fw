//
// Created by Mèir Noordermeer on 29/09/2019.
//

#ifndef RENDERING_FW_VKCONTEXT_SRC_FRAMEBUFFER_HPP
#define RENDERING_FW_VKCONTEXT_SRC_FRAMEBUFFER_HPP

#include <vulkan/vulkan.hpp>

#include "VulkanDevice.h"

namespace vkc
{
class FrameBufferAttachment
{
  public:
	FrameBufferAttachment(VulkanDevice dev) : device(dev) {}
	virtual ~FrameBufferAttachment()
	{
		if (view)
			device->destroyImageView(view);
		if (image)
			device->destroyImage(image);
		if (memory)
			device->freeMemory(memory);
	}

	virtual void cleanup()
	{
		if (view)
			device->destroyImageView(view);
		if (image)
			device->destroyImage(image);
		if (memory)
			device->freeMemory(memory);
	}

	bool hasDepth();
	bool hasStencil();
	bool isDepthStencil();

	VulkanDevice device;
	vk::Image image;
	vk::DeviceMemory memory;
	vk::ImageView view;
	vk::Format format;
	vk::ImageSubresourceRange subresourceRange{};
	vk::AttachmentDescription description{};
};

class FrameBufferSwapChainAttachment : public FrameBufferAttachment
{
  public:
	FrameBufferSwapChainAttachment(VulkanDevice dev, vk::Image swapchainImage, vk::ImageView swapchainImageView,
								   vk::Extent2D ext, vk::Format form)
		: FrameBufferAttachment(dev)
	{
		this->image = swapchainImage;
		this->view = swapchainImageView;
		this->extent = ext;
		this->format = form;

		subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
		subresourceRange.baseArrayLayer = 0;
		subresourceRange.baseMipLevel = 0;
		subresourceRange.layerCount = 1;
		subresourceRange.levelCount = 1;

		description.format = form;
		description.flags = vk::AttachmentDescriptionFlags();
		description.samples = vk::SampleCountFlagBits::e1;
		description.loadOp = vk::AttachmentLoadOp::eClear;
		description.storeOp = vk::AttachmentStoreOp::eStore;
		description.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
		description.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
		description.initialLayout = vk::ImageLayout::eUndefined;
		description.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
	}

	~FrameBufferSwapChainAttachment() override = default;
	void cleanup() override {}

	vk::Extent2D extent;
};

struct AttachmentCreateInfo
{
	AttachmentCreateInfo() = default;
	AttachmentCreateInfo(vk::Extent2D extent, uint32_t attachmentLayerCount, vk::Format attachmentFormat,
						 vk::ImageUsageFlags usageFlags)
		: width(extent.width), height(extent.height), layerCount(attachmentLayerCount), format(attachmentFormat),
		  usage(usageFlags)
	{
	}

	void setExtent(vk::Extent2D extent) { width = extent.width, height = extent.height; }
	void setExtent(uint32_t w, uint32_t h) { width = w, height = h; }
	void setLayerCount(unsigned int count) { layerCount = count; }
	void setFormat(vk::Format f) { format = f; }
	void setUsage(vk::ImageUsageFlags flags) { usage = flags; }

	uint32_t width, height;
	uint32_t layerCount;
	vk::Format format;
	vk::ImageUsageFlags usage;
};

class FrameBuffer
{
  public:
	FrameBuffer(const VulkanDevice &device);
	~FrameBuffer();

	void cleanup();
	uint32_t addSwapChainAttachment(vk::Image swapchainImage, vk::ImageView swapchainView, vk::Extent2D extent,
									vk::Format format);
	uint32_t addAttachment(AttachmentCreateInfo createInfo);
	void createSampler(vk::Filter magFilter, vk::Filter minFilter, vk::SamplerAddressMode addressMode);
	void createRenderPass();

	vk::Framebuffer &getFramebuffer() { return m_Framebuffer; }
	vk::RenderPass &getRenderPass() { return m_RenderPass; }
	vk::Sampler &getSampler() { return m_Sampler; }
	uint32_t get_width() const { return width; }
	uint32_t get_height() const { return height; }
	[[nodiscard]] std::vector<FrameBufferAttachment *> &getAttachments() { return m_Attachments; }
	[[nodiscard]] const std::vector<FrameBufferAttachment *> &getAttachments() const { return m_Attachments; }

	operator vk::Framebuffer() const { return m_Framebuffer; }
	operator vk::RenderPass() const { return m_RenderPass; }
	operator vk::Sampler() const { return m_Sampler; }

  private:
	VulkanDevice m_Device;
	uint32_t width = 0, height = 0;
	vk::Framebuffer m_Framebuffer;
	vk::RenderPass m_RenderPass;
	vk::Sampler m_Sampler;
	std::vector<FrameBufferAttachment *> m_Attachments;
};
} // namespace vkc
#endif // RENDERING_FW_VKCONTEXT_SRC_FRAMEBUFFER_HPP
