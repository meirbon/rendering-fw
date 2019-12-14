#pragma once

#include <string_view>
#include <memory>

#include "RenderSystem.h"
#include "utils/ArrayProxy.h"
#include "utils/Window.h"
#include "utils/gl.h"

#include <ImGuiContext.h>

#define VULKAN_RTX "VulkanRTX"
#define GLRENDERER "GLRenderer"
#define OPTIX6 "OptiX6Context"
#define CPURT "CPURT"

namespace rfw
{
class Application
{
  public:
	static void run(Application *app);

  protected:
	explicit Application(size_t scrWidth, size_t scrHeight, std::string_view title, std::string_view renderAPI);
	~Application();

	Application(const Application &) = delete;
	Application(Application &&) = delete;

	virtual void init() = 0;
	virtual void loadScene(std::unique_ptr<rfw::RenderSystem> &rs) = 0;
	virtual void loadInstances(rfw::utils::ArrayProxy<GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs) = 0;

	virtual void update(std::unique_ptr<rfw::RenderSystem> &rs, float dt) = 0;
	virtual void renderGUI(std::unique_ptr<rfw::RenderSystem> &rs) = 0;

	virtual void cleanup() = 0;

	rfw::RenderStatus status = Reset;
	rfw::Camera camera;
	rfw::utils::Window window;

  private:
	rfw::utils::GLTexture *m_Target = nullptr;
	rfw::utils::GLShader *m_Shader = nullptr;
	imgui::Context m_ImGuiContext;
	std::unique_ptr<rfw::RenderSystem> m_RS;

	void draw();
};

} // namespace rfw