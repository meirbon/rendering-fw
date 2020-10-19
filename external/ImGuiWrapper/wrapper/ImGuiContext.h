#pragma once

#include <imgui.h>
#include <memory>
#include <GLFW/glfw3.h>

namespace imgui
{
class Context
{
  public:
	enum ContextType
	{
		GLFW,
		VULKAN,
	};

	Context(GLFWwindow *window);
	// TODO
	// Context(vk::Instance vulkanInstance, vk::PhysicalDevice pDevice, vk::Device device);

	~Context();

	void newFrame();

	void render();

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