#pragma once

#include <string_view>
#include <memory>

#include "RenderSystem.h"
#include "utils/ArrayProxy.h"
#include "utils/Window.h"
#include "utils/gl.h"

#include <ImGuiContext.h>

static constexpr char *VULKANRTX = "VulkanRTX";
static constexpr char *VULKAN = "VkContext";
static constexpr char *GLRENDERER = "GLRenderer";
static constexpr char *OPTIX6 = "OptiX6Context";
static constexpr char *CPURT = "CPURT";
static constexpr char *EMBREE = "EmbreeRT";

namespace rfw
{
class Application
{
  public:
	static void run(Application *app);

  protected:
	explicit Application(size_t scrWidth, size_t scrHeight, std::string title, std::string renderAPI);
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