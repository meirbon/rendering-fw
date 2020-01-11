#pragma once

namespace vkrtx
{
class Image
{
  public:
	enum Type
	{
		TEXTURE,
		SKYDOME,
		FRAMEBUFFER
	};

	Image(const VulkanDevice &device, vk::ImageType type, vk::Format format, vk::Extent3D extent,
		  vk::ImageTiling tiling, vk::ImageUsageFlags usage, vk::MemoryPropertyFlags memProps,
		  Type allocType = TEXTURE);
	~Image();

	void cleanup();

	template <typename T> bool setData(const std::vector<T> &data, uint32_t width, uint32_t height)
	{
		return setData(data.data(), width, height, sizeof(T));
	}
	bool setData(const void *data, uint32_t width, uint32_t height, uint32_t stride);
	bool createImageView(vk::ImageViewType viewType, vk::Format format, vk::ImageSubresourceRange subresourceRange);
	bool createSampler(vk::Filter magFilter, vk::Filter minFilter, vk::SamplerMipmapMode mipmapMode,
					   vk::SamplerAddressMode addressMode);
	void transitionToLayout(vk::ImageLayout layout, vk::AccessFlags dstAccessMask,
							vk::CommandBuffer cmdBuffer = nullptr);
	vk::DescriptorImageInfo getDescriptorImageInfo() const;

	vk::Extent3D getExtent() const { return m_Extent; }
	vk::Image getImage() const { return m_Image; }
	vk::ImageView getImageView() const { return m_ImageView; }
	vk::Sampler getSampler() const { return m_Sampler; }

	operator vk::Image() const { return m_Image; }
	operator vk::ImageView() const { return m_ImageView; }
	operator vk::Sampler() const { return m_Sampler; }

  private:
	vk::ImageLayout m_CurLayout = vk::ImageLayout::eUndefined;
	VulkanDevice m_Device;
	vk::Extent3D m_Extent;
	vk::Image m_Image = nullptr;
	VmaAllocation m_Allocation;
	VmaAllocationInfo m_AllocInfo;
	vk::ImageView m_ImageView = nullptr;
	vk::Sampler m_Sampler = nullptr;
};
} // namespace vkrtx
