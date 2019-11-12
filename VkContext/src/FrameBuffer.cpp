//
// Created by Mèir Noordermeer on 29/09/2019.
//

#include "FrameBuffer.h"

using namespace vkc;

bool FrameBufferAttachment::hasDepth()
{
	constexpr std::array<vk::Format, 6> formats = {vk::Format::eD16Unorm,		vk::Format::eX8D24UnormPack32,
												   vk::Format::eD32Sfloat,		vk::Format::eD16UnormS8Uint,
												   vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint};

	return std::find(formats.begin(), formats.end(), format) != std::end(formats);
}

bool FrameBufferAttachment::hasStencil()
{
	constexpr std::array<vk::Format, 4> formats = {vk::Format::eS8Uint, vk::Format::eD16UnormS8Uint,
												   vk::Format::eD24UnormS8Uint, vk::Format::eD32SfloatS8Uint};

	return std::find(formats.begin(), formats.end(), format) != std::end(formats);
}

bool FrameBufferAttachment::isDepthStencil() { return hasDepth() || hasStencil(); }

FrameBuffer::FrameBuffer(const Device &device) : m_Device(device) {}

FrameBuffer::~FrameBuffer() { cleanup(); }

void FrameBuffer::cleanup()
{
	for (auto attachment : m_Attachments)
		delete attachment;

	if (m_Sampler)
		m_Device->destroySampler(m_Sampler);
	if (m_RenderPass)
		m_Device->destroyRenderPass(m_RenderPass);
	if (m_Framebuffer)
		m_Device->destroyFramebuffer(m_Framebuffer);
}

uint32_t FrameBuffer::addSwapChainAttachment(vk::Image swapchainImage, vk::ImageView swapchainView, vk::Extent2D extent,
											 vk::Format format)
{
	auto attachment = new FrameBufferSwapChainAttachment(m_Device, swapchainImage, swapchainView, extent, format);
	m_Attachments.push_back(attachment);
	return static_cast<uint32_t>(m_Attachments.size() - 1);
}

uint32_t FrameBuffer::addAttachment(AttachmentCreateInfo createInfo)
{
	auto attachment = new FrameBufferAttachment(m_Device);

	attachment->format = createInfo.format;

	auto aspectMask = vk::ImageAspectFlags();

	if (createInfo.usage & vk::ImageUsageFlagBits::eColorAttachment)
		aspectMask = vk::ImageAspectFlagBits::eColor;

	if (createInfo.usage & vk::ImageUsageFlagBits::eDepthStencilAttachment)
	{
		if (attachment->hasDepth())
			aspectMask = vk::ImageAspectFlagBits::eDepth;
		if (attachment->hasStencil())
			aspectMask = aspectMask | vk::ImageAspectFlagBits::eStencil;
	}

	assert(static_cast<uint32_t>(aspectMask) > 0);

	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setImageType(vk::ImageType::e2D);
	imageCreateInfo.setFormat(createInfo.format);
	imageCreateInfo.setExtent({createInfo.width, createInfo.height, 1});
	imageCreateInfo.setMipLevels(1);
	imageCreateInfo.setArrayLayers(createInfo.layerCount);
	imageCreateInfo.setSamples(vk::SampleCountFlagBits::e1);
	imageCreateInfo.setUsage(createInfo.usage);

	vk::MemoryAllocateInfo memAlloc{};
	vk::MemoryRequirements memReqs{};

	attachment->image = m_Device->createImage(imageCreateInfo);
	memReqs = m_Device->getImageMemoryRequirements(attachment->image);
	memAlloc.allocationSize = memReqs.size;
	memAlloc.memoryTypeIndex = m_Device.getMemoryType(memReqs, vk::MemoryPropertyFlagBits::eDeviceLocal);

	attachment->memory = m_Device->allocateMemory(memAlloc);
	m_Device->bindImageMemory(attachment->image, attachment->memory, 0);

	attachment->subresourceRange.setAspectMask(aspectMask);
	attachment->subresourceRange.setLevelCount(1);
	attachment->subresourceRange.setLayerCount(createInfo.layerCount);

	vk::ImageViewCreateInfo imageViewCreateInfo{};
	imageViewCreateInfo.setViewType(createInfo.layerCount == 1 ? vk::ImageViewType::e2D : vk::ImageViewType::e2DArray);
	imageViewCreateInfo.setFormat(createInfo.format);
	imageViewCreateInfo.subresourceRange = attachment->subresourceRange;

	// TODO: Workaround for depth + stencil attachments
	imageViewCreateInfo.subresourceRange.aspectMask =
		attachment->hasDepth() ? vk::ImageAspectFlagBits::eDepth : aspectMask;
	imageViewCreateInfo.image = attachment->image;

	attachment->view = m_Device->createImageView(imageViewCreateInfo);

	attachment->description.samples = vk::SampleCountFlagBits::e1;
	attachment->description.loadOp = vk::AttachmentLoadOp::eClear;
	attachment->description.storeOp = (createInfo.usage & vk::ImageUsageFlagBits::eSampled)
										 ? vk::AttachmentStoreOp::eStore
										 : vk::AttachmentStoreOp::eDontCare;
	attachment->description.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
	attachment->description.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
	attachment->description.format = createInfo.format;
	attachment->description.initialLayout = vk::ImageLayout::eUndefined;

	if (attachment->isDepthStencil())
		attachment->description.finalLayout = vk::ImageLayout::eDepthStencilReadOnlyOptimal;
	else
		attachment->description.finalLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

	m_Attachments.push_back(attachment);
	return static_cast<uint32_t>(m_Attachments.size() - 1);
}

void FrameBuffer::createSampler(vk::Filter magFilter, vk::Filter minFilter, vk::SamplerAddressMode addressMode)
{
	vk::SamplerCreateInfo createInfo{};
	createInfo.magFilter = magFilter;
	createInfo.minFilter = minFilter;
	createInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
	createInfo.addressModeU = addressMode;
	createInfo.addressModeV = addressMode;
	createInfo.addressModeW = addressMode;
	createInfo.mipLodBias = 0.0f;
	createInfo.maxAnisotropy = 1.0f;
	createInfo.minLod = 0.0f;
	createInfo.maxLod = 1.0f;
	createInfo.borderColor = vk::BorderColor::eFloatOpaqueWhite;
	m_Sampler = m_Device->createSampler(createInfo);
}

void FrameBuffer::createRenderPass()
{
	std::vector<vk::AttachmentDescription> attachmentDescriptions{};
	for (auto &attachment : m_Attachments)
		attachmentDescriptions.push_back(attachment->description);

	std::vector<vk::AttachmentReference> colorRefs;
	vk::AttachmentReference depthRef{};
	bool hasDepth = false;
	bool hasColor = false;

	uint32_t attachmentIdx = 0;

	for (auto attachment : m_Attachments)
	{
		if (attachment->isDepthStencil())
		{
			assert(!hasDepth); // Only 1 depth attachment allowed
			depthRef.attachment = attachmentIdx;
			depthRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
			hasDepth = true;
		}
		else
		{
			colorRefs.emplace_back(attachmentIdx, vk::ImageLayout::eColorAttachmentOptimal);
			hasColor = true;
		}
		attachmentIdx++;
	}

	vk::SubpassDescription subpass{};
	if (hasColor)
	{
		subpass.pColorAttachments = colorRefs.data();
		subpass.colorAttachmentCount = (uint32_t)colorRefs.size();
	}
	if (hasDepth)
	{
		subpass.pDepthStencilAttachment = &depthRef;
	}

	std::array<vk::SubpassDependency, 2> dependencies;

	dependencies[0].srcSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[0].dstSubpass = 0;
	dependencies[0].srcStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
	dependencies[0].dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependencies[0].srcAccessMask = vk::AccessFlagBits::eMemoryRead;
	dependencies[0].dstAccessMask =
		vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
	dependencies[0].dependencyFlags = vk::DependencyFlagBits::eByRegion;

	dependencies[1].srcSubpass = 0;
	dependencies[1].dstSubpass = VK_SUBPASS_EXTERNAL;
	dependencies[1].srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	dependencies[1].dstStageMask = vk::PipelineStageFlagBits::eBottomOfPipe;
	dependencies[1].srcAccessMask =
		vk::AccessFlagBits::eColorAttachmentRead | vk::AccessFlagBits::eColorAttachmentWrite;
	dependencies[1].dstAccessMask = vk::AccessFlagBits::eMemoryRead;
	dependencies[1].dependencyFlags = vk::DependencyFlagBits::eByRegion;

	vk::RenderPassCreateInfo renderPassInfo{};
	renderPassInfo.pAttachments = attachmentDescriptions.data();
	renderPassInfo.attachmentCount = (uint32_t)attachmentDescriptions.size();
	renderPassInfo.subpassCount = 1;
	renderPassInfo.pSubpasses = &subpass;
	renderPassInfo.dependencyCount = 2;
	renderPassInfo.pDependencies = dependencies.data();
	m_RenderPass = m_Device->createRenderPass(renderPassInfo);

	std::vector<vk::ImageView> attachmentViews;
	for (auto &attachment : m_Attachments)
		attachmentViews.push_back(attachment->view);

	uint32_t maxLayers = 0;
	for (auto &attachment : m_Attachments)
	{
		if (attachment->subresourceRange.layerCount > maxLayers)
			maxLayers = attachment->subresourceRange.layerCount;
	}

	vk::FramebufferCreateInfo framebufferInfo{};
	framebufferInfo.renderPass = m_RenderPass;
	framebufferInfo.pAttachments = attachmentViews.data();
	framebufferInfo.attachmentCount = (uint32_t)attachmentViews.size();
	framebufferInfo.width = width;
	framebufferInfo.height = height;
	framebufferInfo.layers = maxLayers;
	m_Framebuffer = m_Device->createFramebuffer(framebufferInfo);
}