#include <rfw/app.h>

#define SKINNED_MESH 0
#define PICA 1
#define PICA_LIGHTS 1
#define SPONZA 0
#define DRAGON 0

using namespace rfw::utils;

class app final : public rfw::app
{
  public:
	app();

  protected:
	void init(std::unique_ptr<rfw::system> &rs) override;
	void load_instances(array_proxy<rfw::geometry_ref> geometry, std::unique_ptr<rfw::system> &rs) override;
	void update(std::unique_ptr<rfw::system> &rs, float dt) override;
	void post_render(std::unique_ptr<rfw::system> &rs) override;
	void cleanup() override;

  private:
	unsigned int mouseX, mouseY;
	rfw::geometry_ref cesiumMan{};
	rfw::instance_ref cesiumManInstance;
	rfw::geometry_ref pica{};
	rfw::instance_ref picaInstance;

	rfw::geometry_ref lightQuad{};
	rfw::instance_ref lightQuadInstance;
	rfw::light_ref pointLight{};
	rfw::light_ref spotLight{};
};

app::app() : rfw::app(512, 512, "RenderingFW", VULKANRTX)
{
	camera = rfw::Camera::deserialize("camera.bin");
	camera.resize(window.get_framebuffer_width(), window.get_framebuffer_height());
	camera.brightness = 0.0f;
	camera.contrast = 0.5f;
	window.add_mouse_pos_callback([this](double x, double y, double, double) {
		mouseX = static_cast<uint>(x * static_cast<double>(window.get_width()));
		mouseY = static_cast<uint>(y * static_cast<double>(window.get_height()));
	});
}

void app::init(std::unique_ptr<rfw::system> &rs)
{
	rs->set_skybox("envmaps/sky_15.hdr");
	cesiumMan = rs->add_object("models/CesiumMan/CesiumMan.gltf", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
	pica = rs->add_object("models/pica/scene.gltf");

	const auto light_material = rs->add_material(vec3(50), 1);

	lightQuad = rs->add_quad(vec3(0, -1, 0), vec3(0, 25, 0), 8.0f, 8.0f, light_material);
	lightQuadInstance = rs->add_instance(lightQuad);
	pointLight = rs->add_point_light(vec3(-15, 10, -5), vec3(10));
	spotLight = rs->add_spot_light(vec3(10, 10, 3), cos(radians(30.0f)), vec3(10), cos(radians(45.0f)), vec3(0, -1, 0));
}

void app::load_instances(array_proxy<rfw::geometry_ref> geometry, std::unique_ptr<rfw::system> &rs)
{
	cesiumManInstance = rs->add_instance(cesiumMan, vec3(1), vec3(10, 0.2f, 3));
	picaInstance = rs->add_instance(pica);

	picaInstance.rotate(180.0f, vec3(0, 1, 0));
	picaInstance.update();
}

void app::update(std::unique_ptr<rfw::system> &rs, float dt)
{
	bool camChanged = false;
	status = window.pressed(KEY_B) ? rfw::Reset : rfw::Converge;
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
	if (window.pressed(key_code::KEY_ESCAPE))
		window.close();
	if (window.pressed(key_code::KEY_W))
		translation.z += 1.0f;
	if (window.pressed(key_code::KEY_S))
		translation.z -= 1.0f;
	if (window.pressed(key_code::KEY_D))
		translation.x += 1.0f;
	if (window.pressed(key_code::KEY_A))
		translation.x -= 1.0f;
	if (window.pressed(key_code::KEY_R))
		translation.y += 1.0f;
	if (window.pressed(key_code::KEY_F))
		translation.y -= 1.0f;
	if (window.pressed(key_code::KEY_UP))
		target.y += 0.001f;
	if (window.pressed(key_code::KEY_DOWN))
		target.y -= 0.001f;
	if (window.pressed(key_code::KEY_RIGHT))
		target.x += 0.001f;
	if (window.pressed(key_code::KEY_LEFT))
		target.x -= 0.001f;

	translation *= dt * 0.01f * (window.pressed(key_code::KEY_LEFT_SHIFT) ? 5.0f : 1.0f);
	target *= dt;

	// Update camera
	if (any(notEqual(translation, vec3(0.0f))))
	{
		camChanged = true;
		camera.translate_relative(translation);
	}
	if (any(notEqual(target, vec3(0.0f))))
	{
		camChanged = true;
		camera.translate_target(target);
	}

	if (window.mousePressed(Mousekey::BUTTON_RIGHT))
		rs->set_probe_index(uvec2(mouseX, mouseY));

	if (camChanged)
		status = rfw::Reset;
}

void app::post_render(std::unique_ptr<rfw::system> &rs) {}

void app::cleanup() { camera.serialize("camera.bin"); }

int main(int argc, char *argv[])
{
	auto application = app();
	rfw::app::run(application);
}
