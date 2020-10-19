#pragma once

#include <string_view>
#include <memory>

#include "system.h"
#include <rfw/utils/array_proxy.h>
#include <rfw/utils/window.h>
#include <rfw/utils/gl.h>

#include <ImGuiContext.h>

#define VULKANRTX "VulkanRTX"
#define VULKAN "VkContext"
#define GLRENDERER "GLRenderer"
#define OPTIX6 "OptiX6Context"
#define CUDART "CUDART"
#define CPURT "CPURT"
#define EMBREE "EmbreeRT"

namespace rfw
{
class app
{
  public:
	static void run(app &);

	app(const app &) = delete;
	app(app &&) = delete;

  protected:
	explicit app(size_t scrWidth, size_t scrHeight, std::string title, std::string renderAPI,
						 bool hidpi = false);
	~app();

	virtual void init(std::unique_ptr<rfw::system> &rs) = 0;
	virtual void load_instances(rfw::utils::array_proxy<geometry_ref> geometry,
								std::unique_ptr<rfw::system> &rs) = 0;

	virtual void update(std::unique_ptr<rfw::system> &rs, float dt) = 0;
	virtual void post_render(std::unique_ptr<rfw::system> &rs) = 0;

	virtual void cleanup() = 0;

	rfw::RenderStatus status = Reset;
	rfw::Camera camera;
	rfw::utils::window window;

	rfw::AvailableRenderSettings renderSettings;
	std::vector<int> settingsCurrentValues;
	std::vector<const char *> settingKeys;
	std::vector<std::vector<const char *>> settingAvailableValues;

  private:
	rfw::utils::texture *m_Target = nullptr;
	rfw::utils::shader *m_Shader = nullptr;
	imgui::Context m_ImGuiContext;
	std::unique_ptr<rfw::system> m_RS;

	void draw();
};

} // namespace rfw