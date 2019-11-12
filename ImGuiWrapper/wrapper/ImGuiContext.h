#pragma once

#include <imgui.h>
#include <memory>
#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#ifdef __APPLE__
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

#ifdef __APPLE__

	void newFrame(MTLRenderPassDescriptor *renderPassDescriptor);

#endif

	void render() const;

	void render(vk::CommandBuffer cmdBuffer);

#ifdef __APPLE__

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