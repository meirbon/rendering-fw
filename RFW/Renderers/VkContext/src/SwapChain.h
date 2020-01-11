//
// Created by Mèir Noordermeer on 12-09-19.
//

#ifndef RENDERING_FW_VKCONTEXT_SRC_SWAPCHAIN_HPP
#define RENDERING_FW_VKCONTEXT_SRC_SWAPCHAIN_HPP

#include <vulkan/vulkan.hpp>

#include <vector>

#include "VulkanDevice.h"

namespace vkc
{

class SwapChain
{
  public:
	struct Buffer
	{
		vk::Image image;
		vk::ImageView view;
	};

	SwapChain(VulkanDevice &device, vk::SurfaceKHR surface,
			  vk::Format desiredFormat = vk::Format::eR8G8B8A8Uint);
	~SwapChain();

	void destroy();
	void create(uint32_t width, uint32_t height, bool vsync = false);

	vk::Result acquireNextImage(vk::Semaphore presentCompleteSemaphore, uint32_t *imageIndex,
								vk::Fence fence = nullptr);
	vk::Result queuePresent(vk::Queue queue, uint32_t imageIndex, vk::Semaphore waitSemaphore = nullptr);
	vk::AttachmentDescription
	getColorAttachmentDescription(vk::SampleCountFlagBits = vk::SampleCountFlagBits::e1,
								  vk::AttachmentLoadOp loadOp = vk::AttachmentLoadOp::eClear,
								  vk::AttachmentStoreOp storeOp = vk::AttachmentStoreOp::eStore,
								  vk::AttachmentLoadOp stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
								  vk::AttachmentStoreOp stencilStoreOp = vk::AttachmentStoreOp::eDontCare) const;

	[[nodiscard]] VulkanDevice getDevice() { return m_Device; }
	[[nodiscard]] vk::SurfaceKHR getSurface() const { return m_Surface; }
	[[nodiscard]] vk::Format getFormat() const { return m_Format; }
	[[nodiscard]] vk::ColorSpaceKHR getColorSpace() const { return m_ColorSpace; }
	[[nodiscard]] vk::SwapchainKHR getSwapChainKHR() const { return m_SwapChain; }
	[[nodiscard]] vk::PresentModeKHR getPresentMode() const { return m_PresentMode; }
	[[nodiscard]] uint32_t getImageCount() const { return m_ImageCount; }
	[[nodiscard]] const std::vector<vk::Image> &getImages() const { return m_Images; }
	[[nodiscard]] const std::vector<SwapChain::Buffer> &getBuffers() const { return m_Buffers; }
	[[nodiscard]] vk::Extent2D getExtent() const { return m_Extent; }

  private:
	VulkanDevice m_Device;
	vk::SurfaceKHR m_Surface;
	vk::Format m_Format{};
	vk::ColorSpaceKHR m_ColorSpace{};
	vk::SwapchainKHR m_SwapChain = nullptr;
	vk::PresentModeKHR m_PresentMode{};
	vk::Extent2D m_Extent;

	uint32_t m_ImageCount = 0;
	std::vector<vk::Image> m_Images;
	std::vector<Buffer> m_Buffers;
};

} // namespace vkc
#endif // RENDERING_FW_VKCONTEXT_SRC_SWAPCHAIN_HPP
