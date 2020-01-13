#include <Application.h>

#define SKINNED_MESH 0
#define PICA 1
#define PICA_LIGHTS 1
#define SPONZA 0
#define DRAGON 0

using namespace rfw;
using namespace utils;

class App : public rfw::Application
{
  public:
	App();

  protected:
	void init(std::unique_ptr<rfw::RenderSystem> &rs) override;
	void load_instances(rfw::utils::ArrayProxy<rfw::GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs) override;
	void update(std::unique_ptr<rfw::RenderSystem> &rs, float dt) override;
	void post_render(std::unique_ptr<rfw::RenderSystem> &rs) override;
	void cleanup() override;

  private:
	unsigned int mouseX, mouseY;
	rfw::GeometryReference cesiumMan{};
	rfw::InstanceReference cesiumManInstance;
	rfw::GeometryReference pica{};
	rfw::InstanceReference picaInstance;

	rfw::GeometryReference lightQuad{};
	rfw::InstanceReference lightQuadInstance;
	rfw::LightReference pointLight{};
	rfw::LightReference spotLight{};
};

App::App() : Application(512, 512, "RenderingFW", VULKANRTX)
{
	camera = rfw::Camera::deserialize("camera.bin");
	camera.resize(window.getFramebufferWidth(), window.getFramebufferHeight());
	camera.brightness = 0.0f;
	camera.contrast = 0.5f;
	window.addMousePosCallback([this](double x, double y, double lastX, double lastY) {
		mouseX = static_cast<uint>(x * double(window.getWidth()));
		mouseY = static_cast<uint>(y * double(window.getHeight()));
	});
}

void App::init(std::unique_ptr<rfw::RenderSystem> &rs)
{
	rs->setSkybox("Envmaps/sky_15.hdr");
	cesiumMan = rs->addObject("Models/CesiumMan/CesiumMan.gltf", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
	pica = rs->addObject("Models/pica/scene.gltf");

	auto lightMaterial = rs->addMaterial(vec3(50), 1);

	lightQuad = rs->addQuad(vec3(0, -1, 0), vec3(0, 25, 0), 8.0f, 8.0f, lightMaterial);
	lightQuadInstance = rs->addInstance(lightQuad);
	pointLight = rs->addPointLight(vec3(-15, 10, -5), vec3(10));
	spotLight = rs->addSpotLight(vec3(10, 10, 3), cos(radians(30.0f)), vec3(10), cos(radians(45.0f)), vec3(0, -1, 0));
}

void App::load_instances(rfw::utils::ArrayProxy<rfw::GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs)
{
	cesiumManInstance = rs->addInstance(cesiumMan, vec3(1), vec3(10, 0.2f, 3));
	picaInstance = rs->addInstance(pica);

	picaInstance.rotate(180.0f, vec3(0, 1, 0));
	picaInstance.update();
}

void App::update(std::unique_ptr<rfw::RenderSystem> &rs, float dt)
{
	bool camChanged = false;
	status = window.pressed(KEY_B) ? Reset : Converge;
	auto translation = vec3(0.0f);
	auto target = vec3(0.0f);

	if (window.pressed(KEY_EQUAL))
	{
		camChanged = true;
		camera.aperture += 0.0001f;
	}
	if (window.pressed(KEY_MINUS))
	{
		camChanged = true;
		camera.aperture -= 0.0001f;
	}

	if (window.pressed(KEY_LEFT_BRACKET))
	{
		camChanged = true;
		camera.focalDistance -= 0.1f;
	}
	if (window.pressed(KEY_RIGHT_BRACKET))
	{
		camChanged = true;
		camera.focalDistance += 0.1f;
	}

	camera.aperture = max(camera.aperture, 0.00001f);
	camera.focalDistance = max(camera.focalDistance, 0.01f);

	// Key handling
	if (window.pressed(Keycode::KEY_ESCAPE))
		window.close();
	if (window.pressed(Keycode::KEY_W))
		translation.z += 1.0f;
	if (window.pressed(Keycode::KEY_S))
		translation.z -= 1.0f;
	if (window.pressed(Keycode::KEY_D))
		translation.x += 1.0f;
	if (window.pressed(Keycode::KEY_A))
		translation.x -= 1.0f;
	if (window.pressed(Keycode::KEY_R))
		translation.y += 1.0f;
	if (window.pressed(Keycode::KEY_F))
		translation.y -= 1.0f;
	if (window.pressed(Keycode::KEY_UP))
		target.y += 0.001f;
	if (window.pressed(Keycode::KEY_DOWN))
		target.y -= 0.001f;
	if (window.pressed(Keycode::KEY_RIGHT))
		target.x += 0.001f;
	if (window.pressed(Keycode::KEY_LEFT))
		target.x -= 0.001f;

	translation *= dt * 0.01f * (window.pressed(Keycode::KEY_LEFT_SHIFT) ? 5.0f : 1.0f);
	target *= dt;

	// Update camera
	if (any(notEqual(translation, vec3(0.0f))))
	{
		camChanged = true;
		camera.translateRelative(translation);
	}
	if (any(notEqual(target, vec3(0.0f))))
	{
		camChanged = true;
		camera.translateTarget(target);
	}

	if (window.mousePressed(Mousekey::BUTTON_RIGHT))
		rs->setProbeIndex(uvec2(mouseX, mouseY));

	if (camChanged)
		status = Reset;
}

void App::post_render(std::unique_ptr<rfw::RenderSystem> &rs){}

void App::cleanup() { camera.serialize("camera.bin"); }

int main(int argc, char *argv[])
{
	auto app = new App();
	rfw::Application::run(app);
}
