#include <rfw/system.h>
#include <rfw/app.h>

#define SKINNED_MESH 1
#define CESIUMMAN 1
#define POLLY 0
#define PICA 1
#define SPONZA 0
#define PICA_LIGHTS 0
#define DRAGON 0

using namespace rfw::utils;

class app final : public rfw::app
{
  public:
	explicit app(const std::string &renderer = "");

  protected:
	void init(std::unique_ptr<rfw::system> &rs) override;
	void load_instances(array_proxy<rfw::geometry_ref> geometry, std::unique_ptr<rfw::system> &rs) override;
	void update(std::unique_ptr<rfw::system> &rs, float dt) override;
	void post_render(std::unique_ptr<rfw::system> &rs) override;
	void cleanup() override;

  private:
	unsigned int mouseX, mouseY;
#if SKINNED_MESH
	rfw::geometry_ref skinnedMesh{};
	rfw::instance_ref skinnedMeshInstance;
#endif
#if CESIUMMAN
	rfw::geometry_ref cesiumMan{};
	rfw::instance_ref cesiumManInstance;
#endif
#if POLLY
	rfw::geometry_ref polly;
	rfw::instance_ref pollyInstance;
#endif
#if PICA
	rfw::geometry_ref pica{};
	rfw::instance_ref picaInstance;
	rfw::geometry_ref lightQuad{};
	rfw::instance_ref lightQuadInstance;
#elif SPONZA
	rfw::geometry_ref sponza;
	rfw::instance_ref sponzaInstance;
	rfw::geometry_ref sponzaAreaLight;
	rfw::instance_ref sponzaAreaLightRef;
#endif
#if DRAGON
	rfw::geometry_ref dragon;
	rfw::instance_ref dragonInstance;
	rfw::geometry_ref plane;
	rfw::instance_ref planeInstance;
#endif
#if PICA_LIGHTS
	rfw::light_ref pointLight;
	rfw::light_ref spotLight;
#endif

	rfw::system::ProbeResult probe = {};
	size_t instanceID = 0, materialID = 0;
	rfw::HostMaterial hostMaterial = {};
	rfw::RenderStats stats = {};

	float distance = 0;
	float speedModifier = 0.2f;

	vec3 hit_point = vec3(0);

	bool updateFocus = false;
	bool camChanged = false;
	bool playAnimations = false;
	bool followFocus = false;
};

app::app(const std::string &renderer) : rfw::app(512, 512, "RenderingFW", !renderer.empty() ? renderer : EMBREE)
{
	camera = rfw::Camera::deserialize("camera.bin");
	camera.resize(window.get_framebuffer_width(), window.get_framebuffer_height());
	window.add_mouse_pos_callback([this](double x, double y, double, double) {
		mouseX = static_cast<uint>(x * static_cast<double>(window.get_width()));
		mouseY = static_cast<uint>(y * static_cast<double>(window.get_height()));
	});
}

void app::init(std::unique_ptr<rfw::system> &rs)
{
	// Initialization of your application
	rs->set_skybox("envmaps/sky_15.hdr");
#if SKINNED_MESH
	skinnedMesh = rs->add_object("models/capture.DAE");
#endif
#if CESIUMMAN
	cesiumMan = rs->add_object("models/CesiumMan/CesiumMan.gltf");
#endif
#if POLLY
	polly = rs->add_object("models/project_polly.glb", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
#endif
#if PICA
	pica = rs->add_object("models/pica/scene.gltf");
	auto lightMaterial = rs->add_material(vec3(10), 1);
	lightQuad = rs->add_quad(vec3(0, -1, 0), vec3(0, 25, 0), 12.0f, 12.0f, lightMaterial);
#elif SPONZA
	sponza = rs->add_object("models/sponza/sponza.obj");
	auto material = rs->add_material(vec3(100), 1.0f);
	sponzaAreaLight = rs->add_quad(vec3(0, -1, 0), vec3(0, 0, 0), 20.0f, 100.0f, material);
	sponzaAreaLightRef = rs->add_instance(sponzaAreaLight, vec3(1), vec3(0, 60.0f, 0), 1.0f, vec3(1.0f));
#endif
#if PICA_LIGHTS
	pointLight = rs->add_point_light(vec3(-15, 10, -5), vec3(20));
	spotLight = rs->add_spot_light(vec3(10, 10, 3), 10.0f, vec3(30), 30.0f, vec3(0, -1, 0));
#endif
#if DRAGON
	const auto dragonMaterial = rs->add_material(vec3(255.f / 255.f, 231.f / 255.f, 102.f / 255.f), 0.05f);
	auto material = rs->get_material(dragonMaterial);
	material.metallic = 1.0f;
	rs->set_material(dragonMaterial, material);
	dragon = rs->add_object("models/dragon.obj", dragonMaterial);
	const auto planeMat = rs->add_material(vec3(1.f, 0.4f, 0.4f), 0.03f);
	material = rs->get_material(planeMat);
	material.metallic = 0.2f;
	rs->set_material(planeMat, material);
	plane = rs->add_quad(vec3(0, 1, 0), vec3(5, 1.47f, -2), 3.0f, 3.0f, planeMat);

	rs->add_point_light(vec3(5, 10, 2), vec3(100));
#endif
}

void app::load_instances(rfw::utils::array_proxy<rfw::geometry_ref> geometry, std::unique_ptr<rfw::system> &rs)
{
#if SKINNED_MESH
	skinnedMeshInstance = rs->add_instance(skinnedMesh, vec3(4));
#endif
#if CESIUMMAN
	cesiumManInstance = rs->add_instance(cesiumMan, vec3(3), vec3(10, 0.2f, 3));
#endif
#if POLLY
	pollyInstance = rs->add_instance(polly, vec3(1), vec3(8, 2, 1));
#endif
#if PICA
	picaInstance = rs->add_instance(pica);
	picaInstance.rotate(180.0f, vec3(0, 1, 0));
	picaInstance.update();
	lightQuadInstance = rs->add_instance(lightQuad);
#elif SPONZA
	sponzaInstance = rs->add_instance(sponza, vec3(0.2f));
#endif
#if DRAGON
	dragonInstance = rs->add_instance(dragon, vec3(1), vec3(5, 1.83f, -2));
	planeInstance = rs->add_instance(plane);
#endif
}

void app::update(std::unique_ptr<rfw::system> &rs, float dt)
{
	stats = rs->get_statistics();
	if (playAnimations)
		rs->set_animations_to(static_cast<float>(glfwGetTime()));

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
	{
		updateFocus = true;
		rs->set_probe_index(uvec2(mouseX, mouseY));
		try
		{
			probe = rs->get_probe_result();
			instanceID = probe.object.get_index();
			materialID = probe.materialIdx;
			hostMaterial = rs->get_material(materialID);
			hit_point = camera.position + probe.distance * camera.direction;
		}
		catch (const std::exception &e)
		{
			WARNING("Exception: %s", e.what());
		}
	}

	camera.aperture = max(camera.aperture, 0.00001f);
	camera.focalDistance = max(camera.focalDistance, 0.01f);

	if (updateFocus)
	{
		updateFocus = false;
		camChanged = true;
		camera.focalDistance = probe.distance;
	}

	if (camChanged)
	{
		status = rfw::Reset;
		camChanged = false;

		if (followFocus)
			camera.focalDistance = length(hit_point - camera.position);
	}

	camera.aperture = max(camera.aperture, 0.00001f);
	camera.focalDistance = max(camera.focalDistance, 0.01f);
}

void app::post_render(std::unique_ptr<rfw::system> &rs)
{
	ImGui::Begin("Stats");
	ImGui::Checkbox("Animations", &playAnimations);
	for (int i = 0, s = static_cast<int>(settingKeys.size()); i < s; i++)
	{
		if (ImGui::ListBox(settingKeys[i], &settingsCurrentValues[i], settingAvailableValues[i].data(),
						   static_cast<int>(settingAvailableValues[i].size())))
			rs->set_setting(rfw::RenderSetting(settingKeys[i], settingAvailableValues[i][settingsCurrentValues[i]]));
	}

	ImGui::Text("# Primary: %6ik (%2.1fM/s)", stats.primaryCount / 1000,
				stats.primaryCount / (max(1.0f, stats.primaryTime * 1000000)));
	ImGui::Text("# Secondary: %6ik (%2.1fM/s)", stats.secondaryCount / 1000,
				stats.secondaryCount / (max(1.0f, stats.secondaryTime * 1000000)));
	ImGui::Text("# Deep: %6ik (%2.1fM/s)", stats.deepCount / 1000,
				stats.deepCount / (max(1.0f, stats.deepTime * 1000000)));
	ImGui::Text("# Shadow: %6ik (%2.1fM/s)", stats.shadowCount / 1000,
				stats.shadowCount / (max(1.0f, stats.shadowTime * 1000000)));

	ImGui::Text("Primary %2.2f ms", stats.primaryTime);
	ImGui::Text("Secondary %2.2f ms", stats.secondaryTime);
	ImGui::Text("Deep %2.2f ms", stats.deepTime);
	ImGui::Text("Shadow %2.2f ms", stats.shadowTime);
	ImGui::Text("Shade %2.2f ms", stats.shadeTime);
	ImGui::Text("Finalize %2.2f ms", stats.finalizeTime);
	ImGui::Text("Animation %2.2f ms", stats.animationTime);
	ImGui::Text("Render %2.2f ms", stats.renderTime);
	ImGui::Text("FPS %4.1f", 1000.0f / stats.renderTime);

	ImGui::Separator();
	ImGui::BeginGroup();
	ImGui::Text("Camera");
	if (ImGui::Button("Reset"))
	{
		followFocus = false;
		camera.reset();
	}

	ImGui::Checkbox("Follow focus", &followFocus);
	ImGui::SliderFloat("Contrast", &camera.contrast, 0.0f, 2.0f, "%.7f");
	ImGui::SliderFloat("Brightness", &camera.brightness, 0.0f, 2.0f, "%.7f");
	camChanged |= ImGui::DragFloat("Aperture", &camera.aperture, 0.0001f, 0.000001f, 1.0f, "%.7f");
	camChanged |= ImGui::DragFloat("Focal dist", &camera.focalDistance, 0.0001f, 0.00001f, 1e10f, "%.7f");
	camChanged |= ImGui::DragFloat("FOV", &camera.FOV, 0.1f, 20.0f, 120.0f);
	ImGui::DragFloat("Speed", &speedModifier, 0.1f, 0.001f, 10000.0f);
	ImGui::EndGroup();

	ImGui::Separator();

	ImGui::BeginGroup();
	ImGui::Text("Probe pixel (%u %u)", rs->get_probe_index().x, rs->get_probe_index().y);
	ImGui::Text("Distance %f", distance);
	ImGui::Text("Instance %zu", instanceID);
	try
	{
		auto instanceObject = rs->get_instance_ref(instanceID);

		auto instanceTranslation = instanceObject.get_translation();
		auto instanceScaling = instanceObject.get_scaling();
		auto instanceRotation = instanceObject.get_rotation();

		ImGui::Text("Instance");
		if (ImGui::DragFloat3("Translation", value_ptr(instanceTranslation), 0.1f, -1e20f, 1e20f))
		{
			instanceObject.set_translation(instanceTranslation);
			instanceObject.update();
		}
		if (ImGui::DragFloat3("Scaling", value_ptr(instanceScaling), 0.1f, -1e20f, 1e20f))
		{
			instanceObject.set_scaling(instanceScaling);
			instanceObject.update();
		}
		if (ImGui::DragFloat3("Rotation", value_ptr(instanceRotation), 0.01f, -1e20f, 1e20f))
		{
			instanceObject.set_rotation(instanceRotation);
			instanceObject.update();
		}
	}
	catch (const std::exception &e)
	{
		WARNING(e.what());
	}
	ImGui::EndGroup();

	ImGui::Separator();

	ImGui::BeginGroup();
	ImGui::Text("Material");
	if (ImGui::Button("Save"))
		rs->set_material(materialID, hostMaterial);
	ImGui::Text("Material %zu", materialID);
	ImGui::ColorEdit3("Color", value_ptr(hostMaterial.color), ImGuiColorEditFlags_DisplayRGB);
	ImGui::SliderFloat3("Absorption", value_ptr(hostMaterial.absorption), 0.0f, 10.0f, "%.0f");
	ImGui::SliderFloat("Ior", &hostMaterial.eta, 1.0f, 2.0f);
	ImGui::SliderFloat("Roughness", &hostMaterial.roughness, 0.0f, 1.0f);
	ImGui::SliderFloat("Transmission", &hostMaterial.transmission, 0.0f, 1.0f);
	ImGui::SliderFloat("Metallic", &hostMaterial.metallic, 0.0f, 1.0f);
	ImGui::SliderFloat("ClearCoat", &hostMaterial.clearcoat, 0.0f, 1.0f);
	ImGui::SliderFloat("ClearCoatGloss", &hostMaterial.clearcoatGloss, 0.0f, 1.0f);
	ImGui::SliderFloat("Sheen", &hostMaterial.sheen, 0.0f, 1.0f);
	ImGui::SliderFloat("Sheen tint", &hostMaterial.sheenTint, 0.0f, 10.f);
	ImGui::SliderFloat("Specular", &hostMaterial.specular, 0.0f, 1.0f);
	ImGui::SliderFloat("Spec tint", &hostMaterial.specularTint, 0.0f, 10.f);
	ImGui::SliderFloat("Subsurface", &hostMaterial.subsurface, 0.0f, 1.0f);
	ImGui::EndGroup();

	ImGui::End();
}

void app::cleanup() { camera.serialize("camera.bin"); }

int main(int argc, char *argv[])
{
	std::string renderer;

	if (argc > 1)
	{
		if (argc == 2)
		{
			const std::string arg = argv[1];
			if (string::begins_with(arg, "-r="))
				renderer = arg.substr(strlen("-r="));
			else if (string::begins_with(arg, "--renderer="))
				renderer = arg.substr(strlen("--renderer="));
		}
		else if (argc >= 3)
		{
			const std::string arg = argv[1];
			if (arg == "-r" || arg == "--renderer")
				renderer = argv[2];
		}
	}

	auto application = app(renderer);
	rfw::app::run(application);
}
