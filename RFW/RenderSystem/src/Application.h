#pragma once

#include <string_view>
#include <memory>

#include "RenderSystem.h"
#include "utils/ArrayProxy.h"
#include "utils/Window.h"
#include "utils/gl.h"

#include <ImGuiContext.h>

#define VULKANRTX "VulkanRTX"
#define VULKAN "VkContext"
#define GLRENDERER "GLRenderer"
#define OPTIX6 "OptiX6Context"
#define CUDART "CUDART"
#define CPURT "CPURT"
#define EMBREE "EmbreeRT"
#define CLRT "CLRT"

namespace rfw
{
class Application
{
  public:
	static void run(Application *app);

	Application(const Application &) = delete;
	Application(Application &&) = delete;

  protected:
	explicit Application(size_t scrWidth, size_t scrHeight, std::string title, std::string renderAPI, bool hidpi = false);
	~Application();

	virtual void init(std::unique_ptr<rfw::RenderSystem> &rs) = 0;
	virtual void load_instances(rfw::utils::ArrayProxy<GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs) = 0;

	virtual void update(std::unique_ptr<rfw::RenderSystem> &rs, float dt) = 0;
	virtual void post_render(std::unique_ptr<rfw::RenderSystem> &rs) = 0;

	virtual void cleanup() = 0;

	rfw::RenderStatus status = Reset;
	rfw::Camera camera;
	rfw::utils::Window window;

	rfw::AvailableRenderSettings renderSettings;
	std::vector<int> settingsCurrentValues;
	std::vector<const char *> settingKeys;
	std::vector<std::vector<const char *>> settingAvailableValues;

  private:
	rfw::utils::GLTexture *m_Target = nullptr;
	rfw::utils::GLShader *m_Shader = nullptr;
	imgui::Context m_ImGuiContext;
	std::unique_ptr<rfw::RenderSystem> m_RS;

	void draw();
};

} // namespace rfw