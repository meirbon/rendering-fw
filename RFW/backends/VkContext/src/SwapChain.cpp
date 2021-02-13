//
// Created by Mèir Noordermeer on 12-09-19.
//

#include "SwapChain.h"
#include "CheckVK.h"

#include <rfw/math.h>

#include <array>

vkc::SwapChain::SwapChain(VulkanDevice &device, vk::SurfaceKHR surface, vk::Format desiredFormat)
	: m_Device(device), m_Surface(surface)
{
	vk::PhysicalDevice pDevice = device.getPhysicalDevice();

	const std::vector<vk::SurfaceFormatKHR> surfaceFormats = pDevice.getSurfaceFormatsKHR(surface);
	if (surfaceFormats.size() == 1 && surfaceFormats.at(0).format == vk::Format::eUndefined)
	{
		m_Format = desiredFormat;
		m_ColorSpace = surfaceFormats.at(0).colorSpace;
	}
	else
	{
		bool found = false;
		for (const auto &sf : surfaceFormats)
		{
			if (sf.format == desiredFormat)
			{
				m_Format = sf.format;
				m_ColorSpace = sf.colorSpace;
				found = true;
				break;
			}
		}

		if (!found)
		{
			m_Format = surfaceFormats.at(0).format;
			m_ColorSpace = surfaceFormats.at(0).colorSpace;
		}
	}
}

vkc::SwapChain::~SwapChain() { destroy(); }

void vkc::SwapChain::destroy()
{
	if (m_SwapChain && !m_Buffers.empty())
	{
		for (auto &bf : m_Buffers)
			m_Device->destroyImageView(bf.view);

		m_Buffers.clear();
		m_Device->destroySwapchainKHR(m_SwapChain);
	}

	m_SwapChain = nullptr;
}

void vkc::SwapChain::create(uint32_t width, uint32_t height, bool vsync)
{
	m_Device->waitIdle();

	assert(width > 0);
	assert(height > 0);

	const vk::SwapchainKHR oldSwapChain = m_SwapChain;
	const vk::PhysicalDevice physicalDevice = m_Device;

	const vk::SurfaceCapabilitiesKHR surfCaps = physicalDevice.getSurfaceCapabilitiesKHR(m_Surface);
	const std::vector<vk::PresentModeKHR> presentModes = physicalDevice.getSurfacePresentModesKHR(m_Surface);

	m_Extent = vk::Extent2D(glm::clamp(width, surfCaps.minImageExtent.width, surfCaps.maxImageExtent.width),
							glm::clamp(height, surfCaps.minImageExtent.height, surfCaps.maxImageExtent.height));
	m_PresentMode = vk::PresentModeKHR::eFifo;

	if (!vsync)
	{
		for (const auto &pm : presentModes)
		{
			if (pm == vk::PresentModeKHR::eMailbox)
			{
				m_PresentMode = pm;
				break;
			}

			if (m_PresentMode != vk::PresentModeKHR::eMailbox && pm == vk::PresentModeKHR::eImmediate)
				m_PresentMode = pm;
		}
	}

	uint32_t desiredNrOfImages = surfCaps.minImageCount + 1;

	// Make sure we do not create too many images
	if (surfCaps.maxImageCount == 0) // Sanity check, work-around for primus vk
		desiredNrOfImages = surfCaps.minImageCount > 1 ? surfCaps.minImageCount : 1;
	else if (surfCaps.minImageCount > 0 && desiredNrOfImages > surfCaps.maxImageCount)
		desiredNrOfImages = surfCaps.maxImageCount;

	vk::SurfaceTransformFlagBitsKHR preTransform{};
	if (surfCaps.supportedTransforms & vk::SurfaceTransformFlagBitsKHR::eIdentity)
		preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity; // We'd rather have a non-rotating transform
	else
		preTransform = surfCaps.currentTransform;

	m_ImageCount = desiredNrOfImages;
	vk::CompositeAlphaFlagBitsKHR compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;

	const std::array<vk::CompositeAlphaFlagBitsKHR, 4> compositeAlphaFlags = {
		vk::CompositeAlphaFlagBitsKHR::eOpaque, vk::CompositeAlphaFlagBitsKHR::ePreMultiplied,
		vk::CompositeAlphaFlagBitsKHR::ePostMultiplied, vk::CompositeAlphaFlagBitsKHR::eInherit};

	for (const auto &cAlphaFlag : compositeAlphaFlags)
	{
		if (surfCaps.supportedCompositeAlpha & cAlphaFlag)
		{
			compositeAlpha = cAlphaFlag;
			break;
		}
	}

	// Create the actual swap chain
	vk::SwapchainCreateInfoKHR createInfo{};
	createInfo.setPNext(nullptr);
	createInfo.setSurface(m_Surface);
	createInfo.setMinImageCount(desiredNrOfImages);
	createInfo.setImageFormat(m_Format);
	createInfo.setImageExtent(m_Extent);
	createInfo.setImageUsage(vk::ImageUsageFlagBits::eColorAttachment);
	createInfo.setPreTransform(preTransform);
	createInfo.setImageArrayLayers(1);
	createInfo.setImageSharingMode(vk::SharingMode::eExclusive);
	createInfo.setQueueFamilyIndexCount(0);
	createInfo.setPQueueFamilyIndices(nullptr);
	createInfo.setOldSwapchain(oldSwapChain);
	createInfo.setClipped(true);
	createInfo.setCompositeAlpha(compositeAlpha);

	if (surfCaps.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferSrc)
		createInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferSrc;
	if (surfCaps.supportedUsageFlags & vk::ImageUsageFlagBits::eTransferDst)
		createInfo.imageUsage |= vk::ImageUsageFlagBits::eTransferDst;

	m_SwapChain = m_Device->createSwapchainKHR(createInfo);
	assert(m_SwapChain);

	if (oldSwapChain)
	{
		for (auto &buf : m_Buffers)
			m_Device->destroyImageView(buf.view);

		m_Buffers.clear();
		m_Device->destroySwapchainKHR(oldSwapChain);
	}

	m_Images = m_Device->getSwapchainImagesKHR(m_SwapChain);
	m_Buffers.resize(m_Images.size());
	for (int i = 0, len = static_cast<int>(m_Buffers.size()); i < len; i++)
	{
		vk::ImageViewCreateInfo imageViewCreateInfo{};
		imageViewCreateInfo.setPNext(nullptr);
		imageViewCreateInfo.setFormat(m_Format);
		imageViewCreateInfo.setComponents(
			{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
		imageViewCreateInfo.subresourceRange.setAspectMask(vk::ImageAspectFlagBits::eColor);
		imageViewCreateInfo.subresourceRange.setBaseMipLevel(0);
		imageViewCreateInfo.subresourceRange.setLevelCount(1);
		imageViewCreateInfo.subresourceRange.setBaseArrayLayer(0);
		imageViewCreateInfo.subresourceRange.setLayerCount(1);
		imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
		imageViewCreateInfo.setFlags(vk::ImageViewCreateFlags());

		m_Buffers.at(i).image = m_Images.at(i);
		imageViewCreateInfo.setImage(m_Images.at(i));
		m_Buffers.at(i).view = m_Device->createImageView(imageViewCreateInfo);
	}

	auto cmdBuffer = m_Device.createOneTimeCmdBuffer();
	for (int i = 0; i < m_Buffers.size(); i++)
	{
		if (i != 0)
			cmdBuffer.begin();
		const auto image = m_Images.at(i);

		const auto imgBarrier =
			vk::ImageMemoryBarrier(vk::AccessFlags(), vk::AccessFlagBits::eColorAttachmentWrite,
								   vk::ImageLayout::eUndefined, vk::ImageLayout::eGeneral, 0, 0, image,
								   vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
		cmdBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eAllCommands, vk::PipelineStageFlagBits::eAllCommands,
								   vk::DependencyFlags(), {}, {}, {imgBarrier});

		cmdBuffer.end();

		cmdBuffer.submit(m_Device.getGraphicsQueue(), true);
	}
}

vk::Result vkc::SwapChain::acquireNextImage(vk::Semaphore presentCompleteSemaphore, uint32_t *imageIndex,
											vk::Fence fence)
{
	assert(m_SwapChain);
	return m_Device->acquireNextImageKHR(m_SwapChain, UINT64_MAX, presentCompleteSemaphore, fence, imageIndex);
}

vk::Result vkc::SwapChain::queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore)
{
	assert(m_SwapChain);

	vk::PresentInfoKHR presentInfo{};
	presentInfo.setPNext(nullptr);
	presentInfo.setSwapchainCount(1);
	presentInfo.setPSwapchains(&m_SwapChain);
	presentInfo.setPImageIndices(&imageIndex);
	presentInfo.setPResults(nullptr);

	if (waitSemaphore)
	{
		presentInfo.setPWaitSemaphores(&waitSemaphore);
		presentInfo.setWaitSemaphoreCount(1);
	}
	else
	{
		presentInfo.setPWaitSemaphores(nullptr);
		presentInfo.setWaitSemaphoreCount(0);
	}

	return queue.presentKHR(&presentInfo);
}

vk::AttachmentDescription vkc::SwapChain::getColorAttachmentDescription(vk::SampleCountFlagBits samples,
																		vk::AttachmentLoadOp loadOp,
																		vk::AttachmentStoreOp storeOp,
																		vk::AttachmentLoadOp stencilLoadOp,
																		vk::AttachmentStoreOp stencilStoreOp) const
{
	vk::AttachmentDescription colorAttachment{};
	colorAttachment.setFormat(m_Format);

	colorAttachment.setSamples(samples);
	colorAttachment.setLoadOp(loadOp);
	colorAttachment.setStoreOp(storeOp);
	colorAttachment.setStencilLoadOp(stencilLoadOp);
	colorAttachment.setStencilStoreOp(stencilStoreOp);
	colorAttachment.setInitialLayout(vk::ImageLayout::eUndefined);
	colorAttachment.setFinalLayout(vk::ImageLayout::ePresentSrcKHR);
	return colorAttachment;
}
