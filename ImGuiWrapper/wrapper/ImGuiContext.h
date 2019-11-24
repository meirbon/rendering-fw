#pragma once

#include <imgui.h>
#include <memory>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#if defined(__APPLE__) && defined(__OBJC__)
#import <Metal/Metal.h>
#include <objc/objc.h>
#endif

namespace imgui
{
class Context
{
  public:
	enum ContextType
	{
		GLFW,
		VULKAN,
		METAL
	};

	Context(GLFWwindow *window);
	// TODO
	// Context(vk::Instance vulkanInstance, vk::PhysicalDevice pDevice, vk::Device device);

	~Context();

	void newFrame();

#if defined(__APPLE__) && defined(__OBJC__)
	void newFrame(MTLRenderPassDescriptor *renderPassDescriptor);
#endif

	void render();

	void render(vk::CommandBuffer cmdBuffer);

#if defined(__APPLE__) && defined(__OBJC__)
	void render(id<MTLCommandBuffer> commandBuffer, id<MTLRenderCommandEncoder> commandEncoder) const;
#endif

  private:
	struct Members
	{
		~Members();

		ImGuiContext *context = nullptr;
		ContextType type{};
	};

	std::shared_ptr<Members> m_Members;
};
} // namespace imgui