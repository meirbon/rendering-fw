//
// Created by meir on 10/25/19.
//

#include <GL/glew.h>
#include <utils.h>
#include <utils/gl/CheckGL.h>

#include "InteropTexture.h"

using namespace vkrtx;

#ifdef WIN32
static PFN_vkGetMemoryWin32HandleKHR getMemoryWin32HandleKHR = nullptr;
#else
static PFN_vkGetMemoryFdKHR getMemoryFdKHR = nullptr;
#endif

InteropTexture::InteropTexture(const VulkanDevice &device, uint32_t width, uint32_t height)
{
	m_Device = device;
	m_Width = width;
	m_Height = height;

	if (glCreateMemoryObjectsEXT == nullptr ||
#ifdef WIN32
		glImportMemoryWin32HandleEXT == nullptr ||
#else // LINUX
		glImportMemoryFdEXT == nullptr ||
#endif
		glTextureStorageMem2DEXT == nullptr)
	{
		FAILURE("A Vulkan-OpenGL interop requires an OpenGL 4.5 context and the following extensions: "
				"GL_EXT_memory_object, GL_EXT_memory_object_fd, GL_EXT_memory_object_win32. At least 1 of these was "
				"not found");
	}

	m_Width = width;
	m_Height = height;

	// Create Vulkan image
	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setPNext(nullptr);
	imageCreateInfo.setArrayLayers(1);
	imageCreateInfo.setExtent({width, height, 1});
	imageCreateInfo.setFlags(vk::ImageCreateFlags());
	imageCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageCreateInfo.setImageType(vk::ImageType::e2D);
	imageCreateInfo.setInitialLayout(vk::ImageLayout::eUndefined);
	imageCreateInfo.setMipLevels(1);
	imageCreateInfo.setQueueFamilyIndexCount(0);
	imageCreateInfo.setPQueueFamilyIndices(nullptr);
	imageCreateInfo.setTiling(vk::ImageTiling());
	imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage);
	m_Image = m_Device->createImage(imageCreateInfo);
	assert(m_Image);

	// Get memory requirements
	auto memoryRequirements = m_Device->getImageMemoryRequirements(m_Image);

	// Allocate memory for image
	vk::ExportMemoryAllocateInfo exportAllocInfo{};
	exportAllocInfo.setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	vk::MemoryAllocateInfo memAllocInfo{};
	memAllocInfo.pNext = &exportAllocInfo;
	const auto unaligned = vk::DeviceSize(memoryRequirements.size * 1.3f);
	m_BufferSize = unaligned + (unaligned % memoryRequirements.alignment);
	memAllocInfo.setAllocationSize(m_BufferSize);
	memoryRequirements.size = m_BufferSize;
	memAllocInfo.setMemoryTypeIndex(
		m_Device.getMemoryType(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal));
	m_Memory = m_Device->allocateMemory(memAllocInfo);
	assert(m_Memory);

	// bind memory to Vulkan image
	m_Device->bindImageMemory(m_Image, m_Memory, 0);

	vk::ImageViewCreateInfo imageViewCreateInfo = {};
	imageViewCreateInfo.setPNext(nullptr);
	imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
	imageViewCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageViewCreateInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	imageViewCreateInfo.setImage(m_Image);
	imageViewCreateInfo.setFlags(vk::ImageViewCreateFlags());
	imageViewCreateInfo.setComponents(
		{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
	m_ImageView = m_Device->createImageView(imageViewCreateInfo);

#ifdef WIN32
	HANDLE handle = INVALID_HANDLE_VALUE;
#else
	int handle = 0;
#endif

#ifdef WIN32 // WINDOWS
	// Resolve extension function if needed
	if (getMemoryWin32HandleKHR == nullptr)
		getMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
			vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryWin32HandleKHR"));
	assert(getMemoryWin32HandleKHR != nullptr);

	// Acquire WIN32 handle to Vulkan initialized memory
	vk::MemoryGetWin32HandleInfoKHR getMemoryHandleInfo =
		vk::MemoryGetWin32HandleInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	getMemoryWin32HandleKHR(m_Device.getVkDevice(), (VkMemoryGetWin32HandleInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != INVALID_HANDLE_VALUE && handle != nullptr);
#else // LINUX
	// Resolve extension function if needed
	if (getMemoryFdKHR == nullptr)
		getMemoryFdKHR =
			reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryFdKHR"));
	assert(getMemoryFdKHR != nullptr);

	// Acquire Fd handle to Vulkan initialized memory
	vk::MemoryGetFdInfoKHR getMemoryHandleInfo =
		vk::MemoryGetFdInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
	getMemoryFdKHR(m_Device.getVkDevice(), (VkMemoryGetFdInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != 0);
#endif

	// Create a new texture object in OpenGL
	glCreateTextures(GL_TEXTURE_2D, 1, &m_TexID);

	// Create external memory object
	glCreateMemoryObjectsEXT(1, &m_GLMemoryObj);
	// Import Vulkan memory handle into OpenGL memory object
#ifdef WIN32
	glImportMemoryWin32HandleEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
	glImportMemoryFdEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handle);
#endif
	CheckGL();

	// Point texture object to external OpenGL memory object
	m_Width = width;
	m_Height = height;
	glTextureStorageMem2DEXT(m_TexID, 1, GL_RGBA32F, m_Width, m_Height, m_GLMemoryObj, 0);
	// Check for any errors
	CheckGL();
}

InteropTexture::InteropTexture(const VulkanDevice &device, uint32_t texID, uint32_t width, uint32_t height)
{
	m_TexID = texID;
	m_Device = device;
	m_Width = width;
	m_Height = height;

	if (glCreateMemoryObjectsEXT == nullptr ||
#ifdef WIN32
		glImportMemoryWin32HandleEXT == nullptr ||
#else // LINUX
		glImportMemoryFdEXT == nullptr ||
#endif
		glTextureStorageMem2DEXT == nullptr)
	{
		FAILURE("A Vulkan-OpenGL interop requires an OpenGL 4.5 context and the following extensions: "
				"GL_EXT_memory_object, GL_EXT_memory_object_fd, GL_EXT_memory_object_win32. At least 1 of these was "
				"not found");
	}

	m_Width = width;
	m_Height = height;

	// Create Vulkan image
	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setPNext(nullptr);
	imageCreateInfo.setArrayLayers(1);
	imageCreateInfo.setExtent({width, height, 1});
	imageCreateInfo.setFlags(vk::ImageCreateFlags());
	imageCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageCreateInfo.setImageType(vk::ImageType::e2D);
	imageCreateInfo.setInitialLayout(vk::ImageLayout::eUndefined);
	imageCreateInfo.setMipLevels(1);
	imageCreateInfo.setQueueFamilyIndexCount(0);
	imageCreateInfo.setPQueueFamilyIndices(nullptr);
	imageCreateInfo.setTiling(vk::ImageTiling());
	imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage);
	m_Image = m_Device->createImage(imageCreateInfo);
	assert(m_Image);

	// Get memory requirements
	auto memoryRequirements = m_Device->getImageMemoryRequirements(m_Image);

	// Allocate memory for image
	vk::ExportMemoryAllocateInfo exportAllocInfo{};
	exportAllocInfo.setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	vk::MemoryAllocateInfo memAllocInfo{};
	memAllocInfo.pNext = &exportAllocInfo;
	const auto unaligned = vk::DeviceSize(memoryRequirements.size * 1.3f);
	m_BufferSize = unaligned + (unaligned % memoryRequirements.alignment);
	memAllocInfo.setAllocationSize(m_BufferSize);
	memoryRequirements.size = m_BufferSize;
	memAllocInfo.setMemoryTypeIndex(
		m_Device.getMemoryType(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal));
	m_Memory = m_Device->allocateMemory(memAllocInfo);
	assert(m_Memory);

	// bind memory to Vulkan image
	m_Device->bindImageMemory(m_Image, m_Memory, 0);

	vk::ImageViewCreateInfo imageViewCreateInfo = {};
	imageViewCreateInfo.setPNext(nullptr);
	imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
	imageViewCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageViewCreateInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	imageViewCreateInfo.setImage(m_Image);
	imageViewCreateInfo.setFlags(vk::ImageViewCreateFlags());
	imageViewCreateInfo.setComponents(
		{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
	m_ImageView = m_Device->createImageView(imageViewCreateInfo);

#ifdef WIN32
	HANDLE handle = INVALID_HANDLE_VALUE;
#else
	int handle = 0;
#endif

#ifdef WIN32 // WINDOWS
	// Resolve extension function if needed
	if (getMemoryWin32HandleKHR == nullptr)
		getMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
			vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryWin32HandleKHR"));
	assert(getMemoryWin32HandleKHR != nullptr);

	// Acquire WIN32 handle to Vulkan initialized memory
	vk::MemoryGetWin32HandleInfoKHR getMemoryHandleInfo =
		vk::MemoryGetWin32HandleInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	getMemoryWin32HandleKHR(m_Device.getVkDevice(), (VkMemoryGetWin32HandleInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != INVALID_HANDLE_VALUE && handle != nullptr);
#else // LINUX
	// Resolve extension function if needed
	if (getMemoryFdKHR == nullptr)
		getMemoryFdKHR =
			reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryFdKHR"));
	assert(getMemoryFdKHR != nullptr);

	// Acquire Fd handle to Vulkan initialized memory
	vk::MemoryGetFdInfoKHR getMemoryHandleInfo =
		vk::MemoryGetFdInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
	getMemoryFdKHR(m_Device.getVkDevice(), (VkMemoryGetFdInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != 0);
#endif

	// Create external memory object
	glCreateMemoryObjectsEXT(1, &m_GLMemoryObj);
	// Import Vulkan memory handle into OpenGL memory object
#if _WIN32
	glImportMemoryWin32HandleEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
	glImportMemoryFdEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handle);
#endif
	CheckGL();

	// Point texture object to external OpenGL memory object
	m_Width = width;
	m_Height = height;
	glTextureStorageMem2DEXT(m_TexID, 1, GL_RGBA32F, m_Width, m_Height, m_GLMemoryObj, 0);
	// Check for any errors
	CheckGL();
}

InteropTexture::~InteropTexture() { cleanup(); }

void InteropTexture::recordTransitionToVulkan(vk::CommandBuffer cmdBuffer)
{
	// Our image has 1 layer
	vk::ImageSubresourceRange subresourceRange{};
	subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;

	// Transition from color attachment to transfer destination
	vk::ImageMemoryBarrier imageMemoryBarrier{};
	imageMemoryBarrier.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
	imageMemoryBarrier.newLayout = vk::ImageLayout::eGeneral;
	imageMemoryBarrier.image = m_Image;
	imageMemoryBarrier.subresourceRange = subresourceRange;
	imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentRead;
	imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	vk::PipelineStageFlags destStageMask = vk::PipelineStageFlagBits::eAllCommands;
	cmdBuffer.pipelineBarrier(srcStageMask, destStageMask, vk::DependencyFlags(), nullptr, nullptr, imageMemoryBarrier);
}

void InteropTexture::recordTransitionToGL(vk::CommandBuffer cmdBuffer)
{
	// Our image has 1 layer
	vk::ImageSubresourceRange subresourceRange;
	subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;

	// Transition our image to be used as a color attachment
	vk::ImageMemoryBarrier imageMemoryBarrier;
	imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
	imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
	imageMemoryBarrier.image = m_Image;
	imageMemoryBarrier.subresourceRange = subresourceRange;
	imageMemoryBarrier.srcAccessMask = vk::AccessFlags();
	imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eColorAttachmentRead;
	vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eTopOfPipe;
	vk::PipelineStageFlags destStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	cmdBuffer.pipelineBarrier(srcStageMask, destStageMask, vk::DependencyFlags(), nullptr, nullptr, imageMemoryBarrier);
}

void InteropTexture::cleanup()
{
	glFlush();
	glFinish();

	if (m_ImageView)
	{
		m_Device->destroyImageView(m_ImageView);
		m_ImageView = nullptr;
	}

	if (m_Image)
	{
		m_Device->destroyImage(m_Image);
		m_Image = nullptr;
	}

	if (m_Memory)
	{
		glDeleteMemoryObjectsEXT(1, &m_GLMemoryObj);
		m_GLMemoryObj = 0;
		m_Device->freeMemory(m_Memory);
		m_Memory = nullptr;
	}
}

std::vector<const char *> InteropTexture::getRequiredExtensions()
{
#ifdef WIN32 // WINDOWS
	return {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
			VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME};
#else // LINUX
	return {VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
			VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME};
#endif
}

void InteropTexture::resize(uint32_t width, uint32_t height, bool deleteOldGLTexture)
{
	// Create a new texture object in OpenGL
	if (deleteOldGLTexture)
		glDeleteTextures(1, &m_TexID);
	glGenTextures(1, &m_TexID);
	assert(m_TexID != 0);

	m_Width = width;
	m_Height = height;

	if (m_ImageView)
		m_Device->destroyImageView(m_ImageView);

	if (m_Image)
		m_Device->destroyImage(m_Image);

	// Create Vulkan image
	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setPNext(nullptr);
	imageCreateInfo.setArrayLayers(1);
	imageCreateInfo.setExtent({width, height, 1});
	imageCreateInfo.setFlags(vk::ImageCreateFlags());
	imageCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageCreateInfo.setImageType(vk::ImageType::e2D);
	imageCreateInfo.setInitialLayout(vk::ImageLayout::eUndefined);
	imageCreateInfo.setMipLevels(1);
	imageCreateInfo.setQueueFamilyIndexCount(0);
	imageCreateInfo.setPQueueFamilyIndices(nullptr);
	imageCreateInfo.setTiling(vk::ImageTiling());
	imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage);
	m_Image = m_Device->createImage(imageCreateInfo);
	assert(m_Image);

	// Get memory requirements
	auto memoryRequirements = m_Device->getImageMemoryRequirements(m_Image);
	if (memoryRequirements.size > m_BufferSize)
	{
		m_Device->freeMemory(m_Memory);

		const auto newSize = vk::DeviceSize(memoryRequirements.size * 1.3f);
		m_BufferSize = newSize + (newSize % memoryRequirements.alignment);
		memoryRequirements.size = m_BufferSize;
		// Allocate memory
		vk::ExportMemoryAllocateInfo exportAllocInfo{};
		exportAllocInfo.setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
		vk::MemoryAllocateInfo memAllocInfo{};
		memAllocInfo.pNext = &exportAllocInfo;
		memAllocInfo.setAllocationSize(m_BufferSize);
		memAllocInfo.setMemoryTypeIndex(
			m_Device.getMemoryType(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal));
		m_Memory = m_Device->allocateMemory(memAllocInfo);
	}
	assert(m_Memory);
	m_Device->bindImageMemory(m_Image, m_Memory, 0);

	vk::ImageViewCreateInfo imageViewCreateInfo = {};
	imageViewCreateInfo.setPNext(nullptr);
	imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
	imageViewCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageViewCreateInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	imageViewCreateInfo.setImage(m_Image);
	imageViewCreateInfo.setFlags(vk::ImageViewCreateFlags());
	imageViewCreateInfo.setComponents(
		{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
	m_ImageView = m_Device->createImageView(imageViewCreateInfo);

#ifdef WIN32
	HANDLE handle = INVALID_HANDLE_VALUE;
#else
	int handle = 0;
#endif

#if _WIN32 // WINDOWS
	// Resolve extension function if needed
	if (getMemoryWin32HandleKHR == nullptr)
		getMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
			vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryWin32HandleKHR"));
	assert(getMemoryWin32HandleKHR != nullptr);

	// Acquire WIN32 handle to Vulkan initialized memory
	vk::MemoryGetWin32HandleInfoKHR getMemoryHandleInfo =
		vk::MemoryGetWin32HandleInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	getMemoryWin32HandleKHR(m_Device.getVkDevice(), (VkMemoryGetWin32HandleInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != INVALID_HANDLE_VALUE && handle != nullptr);
#else // LINUX
	// Resolve extension function if needed
	if (getMemoryFdKHR == nullptr)
		getMemoryFdKHR =
			reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryFdKHR"));
	assert(getMemoryFdKHR != nullptr);

	// Acquire Fd handle to Vulkan initialized memory
	vk::MemoryGetFdInfoKHR getMemoryHandleInfo =
		vk::MemoryGetFdInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
	getMemoryFdKHR(m_Device.getVkDevice(), (VkMemoryGetFdInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != 0);
#endif

	glDeleteMemoryObjectsEXT(1, &m_GLMemoryObj);

	glCreateMemoryObjectsEXT(1, &m_GLMemoryObj);

	assert(m_TexID);
	assert(m_GLMemoryObj);

	// bind Vulkan memory handle to OpenGL memory object
#if _WIN32
	glImportMemoryWin32HandleEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
	glImportMemoryFdEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handle);
#endif

	CheckGL();
	// Point texture object to external OpenGL memory object
	glTextureStorageMem2DEXT(m_TexID, 1, GL_RGBA32F, width, height, m_GLMemoryObj, 0);
	CheckGL();

	auto cmdBuffer = m_Device.createOneTimeCmdBuffer();
	auto queue = m_Device.getGraphicsQueue();

	transitionImageToInitialState(cmdBuffer.getVkCommandBuffer(), queue);
	cmdBuffer.submit(queue, true);
}

void InteropTexture::resize(GLuint ID, uint32_t width, uint32_t height, bool deleteOldGLTexture)
{
	m_TexID = ID;
	m_Width = width;
	m_Height = height;

	if (m_ImageView)
		m_Device->destroyImageView(m_ImageView);

	if (m_Image)
		m_Device->destroyImage(m_Image);

	// Create Vulkan image
	vk::ImageCreateInfo imageCreateInfo{};
	imageCreateInfo.setPNext(nullptr);
	imageCreateInfo.setArrayLayers(1);
	imageCreateInfo.setExtent({width, height, 1});
	imageCreateInfo.setFlags(vk::ImageCreateFlags());
	imageCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageCreateInfo.setImageType(vk::ImageType::e2D);
	imageCreateInfo.setInitialLayout(vk::ImageLayout::eUndefined);
	imageCreateInfo.setMipLevels(1);
	imageCreateInfo.setQueueFamilyIndexCount(0);
	imageCreateInfo.setPQueueFamilyIndices(nullptr);
	imageCreateInfo.setTiling(vk::ImageTiling());
	imageCreateInfo.setUsage(vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage);
	m_Image = m_Device->createImage(imageCreateInfo);
	assert(m_Image);

	// Get memory requirements
	auto memoryRequirements = m_Device->getImageMemoryRequirements(m_Image);
	if (memoryRequirements.size > m_BufferSize)
	{
		m_Device->freeMemory(m_Memory);

		const auto newSize = vk::DeviceSize(memoryRequirements.size * 1.3f);
		m_BufferSize = newSize + (newSize % memoryRequirements.alignment);
		memoryRequirements.size = m_BufferSize;
		// Allocate memory
		vk::ExportMemoryAllocateInfo exportAllocInfo{};
		exportAllocInfo.setHandleTypes(vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
		vk::MemoryAllocateInfo memAllocInfo{};
		memAllocInfo.pNext = &exportAllocInfo;
		memAllocInfo.setAllocationSize(m_BufferSize);
		memAllocInfo.setMemoryTypeIndex(
			m_Device.getMemoryType(memoryRequirements, vk::MemoryPropertyFlagBits::eDeviceLocal));
		m_Memory = m_Device->allocateMemory(memAllocInfo);
	}
	assert(m_Memory);
	m_Device->bindImageMemory(m_Image, m_Memory, 0);

	vk::ImageViewCreateInfo imageViewCreateInfo = {};
	imageViewCreateInfo.setPNext(nullptr);
	imageViewCreateInfo.setViewType(vk::ImageViewType::e2D);
	imageViewCreateInfo.setFormat(vk::Format::eR32G32B32A32Sfloat);
	imageViewCreateInfo.setSubresourceRange(vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
	imageViewCreateInfo.setImage(m_Image);
	imageViewCreateInfo.setFlags(vk::ImageViewCreateFlags());
	imageViewCreateInfo.setComponents(
		{vk::ComponentSwizzle::eR, vk::ComponentSwizzle::eG, vk::ComponentSwizzle::eB, vk::ComponentSwizzle::eA});
	m_ImageView = m_Device->createImageView(imageViewCreateInfo);

#ifdef WIN32
	HANDLE handle = INVALID_HANDLE_VALUE;
#else
	int handle = 0;
#endif

#if _WIN32 // WINDOWS
	// Resolve extension function if needed
	if (getMemoryWin32HandleKHR == nullptr)
		getMemoryWin32HandleKHR = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(
			vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryWin32HandleKHR"));
	assert(getMemoryWin32HandleKHR != nullptr);

	// Acquire WIN32 handle to Vulkan initialized memory
	vk::MemoryGetWin32HandleInfoKHR getMemoryHandleInfo =
		vk::MemoryGetWin32HandleInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32KHR);
	getMemoryWin32HandleKHR(m_Device.getVkDevice(), (VkMemoryGetWin32HandleInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != INVALID_HANDLE_VALUE && handle != nullptr);
#else // LINUX
	// Resolve extension function if needed
	if (getMemoryFdKHR == nullptr)
		getMemoryFdKHR =
			reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(m_Device.getVkDevice(), "vkGetMemoryFdKHR"));
	assert(getMemoryFdKHR != nullptr);

	// Acquire Fd handle to Vulkan initialized memory
	vk::MemoryGetFdInfoKHR getMemoryHandleInfo =
		vk::MemoryGetFdInfoKHR(m_Memory, vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);
	getMemoryFdKHR(m_Device.getVkDevice(), (VkMemoryGetFdInfoKHR *)&getMemoryHandleInfo, &handle);
	assert(handle != 0);
#endif

	glDeleteMemoryObjectsEXT(1, &m_GLMemoryObj);

	glCreateMemoryObjectsEXT(1, &m_GLMemoryObj);

	assert(m_TexID);
	assert(m_GLMemoryObj);

	// bind Vulkan memory handle to OpenGL memory object
#if _WIN32
	glImportMemoryWin32HandleEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_WIN32_EXT, handle);
#else
	glImportMemoryFdEXT(m_GLMemoryObj, memoryRequirements.size, GL_HANDLE_TYPE_OPAQUE_FD_EXT, handle);
#endif

	CheckGL();
	// Point texture object to external OpenGL memory object
	glTextureStorageMem2DEXT(m_TexID, 1, GL_RGBA32F, width, height, m_GLMemoryObj, 0);
	CheckGL();

	auto cmdBuffer = m_Device.createOneTimeCmdBuffer();
	auto queue = m_Device.getGraphicsQueue();

	transitionImageToInitialState(cmdBuffer.getVkCommandBuffer(), queue);
	cmdBuffer.submit(queue, true);
}

vk::DescriptorImageInfo vkrtx::InteropTexture::getDescriptorImageInfo() const
{
	return vk::DescriptorImageInfo(nullptr, m_ImageView, vk::ImageLayout::eColorAttachmentOptimal);
}

void InteropTexture::transitionImageToInitialState(vk::CommandBuffer &cmdBuffer, vk::Queue &queue)
{
	vk::ImageSubresourceRange subresourceRange;
	subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
	subresourceRange.levelCount = 1;
	subresourceRange.layerCount = 1;

	vk::ImageMemoryBarrier imageMemoryBarrier;
	imageMemoryBarrier.oldLayout = vk::ImageLayout::eUndefined;
	imageMemoryBarrier.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
	imageMemoryBarrier.image = m_Image;
	imageMemoryBarrier.subresourceRange = subresourceRange;
	imageMemoryBarrier.srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
	imageMemoryBarrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;
	vk::PipelineStageFlags srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput;
	vk::PipelineStageFlags destStageMask = vk::PipelineStageFlagBits::eTransfer;
	cmdBuffer.pipelineBarrier(srcStageMask, destStageMask, vk::DependencyFlags(), nullptr, nullptr, imageMemoryBarrier);
}
