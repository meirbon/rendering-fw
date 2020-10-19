#pragma once

namespace vkrtx
{
class InteropTexture
{
  public:
	InteropTexture(const VulkanDevice &device, uint32_t width, uint32_t height);
	InteropTexture(const VulkanDevice &device, uint32_t texID, uint32_t width, uint32_t height);
	~InteropTexture();

	void recordTransitionToVulkan(vk::CommandBuffer cmdBuffer);
	void recordTransitionToGL(vk::CommandBuffer cmdBuffer);

	void transitionImageToInitialState(vk::CommandBuffer cmdBuffer, vk::Queue &queue);
	void cleanup();

	[[nodiscard]] vk::ImageView getImageView() const { return m_ImageView; }
	[[nodiscard]] vk::Image getImage() const { return m_Image; }
	[[nodiscard]] vk::DeviceMemory getMemory() const { return m_Memory; }
	[[nodiscard]] vk::DeviceSize getBufferSize() const { return m_Width * m_Height * 4 * sizeof(float); }
	[[nodiscard]] uint32_t get_width() const { return m_Width; }
	[[nodiscard]] uint32_t get_height() const { return m_Height; }
	[[nodiscard]] uint32_t getID() const { return m_TexID; }
	static std::vector<const char *> getRequiredExtensions();
	void resize(uint32_t width, uint32_t height, bool deleteOldGLTexture = false);
	void resize(GLuint ID, uint32_t width, uint32_t height, bool deleteOldGLTexture = false);
	vk::DescriptorImageInfo getDescriptorImageInfo() const;

	operator vk::Image() const { return m_Image; }

  private:
	vk::Image m_Image = nullptr;
	vk::DeviceMemory m_Memory = nullptr;
	vk::ImageView m_ImageView = nullptr;
	VulkanDevice m_Device;
	vk::DeviceSize m_BufferSize = 0;
	GLuint m_TexID = 0;
	uint32_t m_GLMemoryObj = 0;
	uint32_t m_Width = 0, m_Height = 0;
};
} // namespace vkrtx
