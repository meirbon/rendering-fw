#include "ImGuiContext.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_impl_vulkan.h"

using namespace imgui;

Context::Context(GLFWwindow *window)
{
	IMGUI_CHECKVERSION();
	m_Members = std::make_shared<Members>();
	m_Members->context = ImGui::CreateContext();
	ImGui::StyleColorsDark();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");
}

//Context::Context(vk::Instance vulkanInstance)
//{
//	IMGUI_CHECKVERSION();
//	m_Members = std::make_shared<Members>();
//	m_Members->context = ImGui::CreateContext();
//	// TODO
//}

Context::~Context() = default;

void Context::newFrame()
{
	switch (m_Members->type)
	{
	case (GLFW):
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		break;
	case (VULKAN):
		ImGui_ImplVulkan_NewFrame();
		break;
	default:
		break;
	}

	ImGui::NewFrame();
}

void Context::render() const
{
	ImGui::Render();

	switch (m_Members->type)
	{
	case (GLFW):
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
		break;
	default:
		break;
	}
}

void Context::render(vk::CommandBuffer cmdBuffer)
{
	assert(m_Members->type == VULKAN);
	ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmdBuffer);
}

Context::Members::~Members()
{
	switch (type)
	{
	case (GLFW):
		ImGui_ImplOpenGL3_Shutdown();
		ImGui_ImplGlfw_Shutdown();
		break;
#ifdef __APPLE__
	case (METAL):
		ImGui_ImplMetal_Shutdown();
		break;
#endif
	case (VULKAN):
		ImGui_ImplVulkan_Shutdown();
		break;
	}

	if (context)
		ImGui::DestroyContext(context);
}
