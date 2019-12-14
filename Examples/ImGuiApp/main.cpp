#include <Application.h>

#define SKINNED_MESH 0
#define CESIUMMAN 0
#define POLLY 0
#define PICA 0
#define PICA_LIGHTS 0
#define DRAGON 1
#define SPONZA 0
#define ANIMATE_DRAGON 0

using namespace rfw;
using namespace utils;

class App : public rfw::Application
{
  public:
	App();

  protected:
	void init() override;
	void loadScene(std::unique_ptr<rfw::RenderSystem> &rs) override;
	void loadInstances(rfw::utils::ArrayProxy<rfw::GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs) override;
	void update(std::unique_ptr<rfw::RenderSystem> &rs, float dt) override;
	void renderGUI(std::unique_ptr<rfw::RenderSystem> &rs) override;
	void cleanup() override;

  private:
	unsigned int mouseX, mouseY;
#if CESIUMMAN
	rfw::GeometryReference cesiumMan;
	rfw::InstanceReference cesiumManInstance;
#endif
#if POLLY
	rfw::GeometryReference polly;
	rfw::InstanceReference pollyInstance;
#endif
#if PICA
	rfw::GeometryReference pica;
	rfw::InstanceReference picaInstance;
#endif
#if DRAGON
	rfw::GeometryReference dragon;
	rfw::InstanceReference dragonInstance;
	rfw::GeometryReference plane;
	rfw::InstanceReference planeInstance;
#endif
#if PICA_LIGHTS
	rfw::GeometryReference lightQuad;
	rfw::InstanceReference lightQuadInstance;
	rfw::LightReference pointLight;
	rfw::LightReference spotLight;
#endif

	rfw::RenderSystem::ProbeResult probe = {};
	size_t instanceID = 0, materialID = 0;
	rfw::HostMaterial hostMaterial = {};
	rfw::RenderStats stats = {};

	float distance = 0;
	float speedModifier = 0.2f;

	bool updateFocus = false;
	bool camChanged = false;
	bool playAnimations = false;
};

App::App() : Application(1280, 720, "RenderingFW", VULKAN_RTX)
{
	camera = rfw::Camera::deserialize("camera.bin");
	camera.resize(window.getWidth(), window.getHeight());
	window.addMousePosCallback([this](double x, double y, double lastX, double lastY) {
		mouseX = static_cast<uint>(x * double(window.getWidth()));
		mouseY = static_cast<uint>(y * double(window.getHeight()));
	});
}

void App::init() {}

void App::loadScene(std::unique_ptr<rfw::RenderSystem> &rs)
{
	rs->setSkybox("Envmaps/sky_15.hdr");
#if CESIUMMAN
	cesiumMan = rs->addObject("Models/CesiumMan.glb", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
#endif
#if POLLY
	polly = rs->addObject("Models/project_polly.glb", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
#endif
#if PICA
	pica = rs->addObject("Models/pica/scene.gltf");
#endif
#if PICA_LIGHTS
	auto lightMaterial = rs->addMaterial(vec3(50), 1);
	lightQuad = rs->addQuad(vec3(0, -1, 0), vec3(0, 25, 0), 8.0f, 8.0f, lightMaterial);
	pointLight = rs->addPointLight(vec3(-15, 10, -5), vec3(10));
	spotLight = rs->addSpotLight(vec3(10, 10, 3), cos(radians(30.0f)), vec3(10), cos(radians(45.0f)), vec3(0, -1, 0));
#endif
#if DRAGON
	const auto dragonMaterial = rs->addMaterial(vec3(255.f / 255.f, 231.f / 255.f, 102.f / 255.f), 0.05f);
	auto material = rs->getMaterial(dragonMaterial);
	material.metallic = 1.0f;
	rs->setMaterial(dragonMaterial, material);
	dragon = rs->addObject("Models/dragon.obj", dragonMaterial);
	const auto planeMat = rs->addMaterial(vec3(1.f, 0.4f, 0.4f), 0.03f);
	material = rs->getMaterial(planeMat);
	material.metallic = 0.2f;
	rs->setMaterial(planeMat, material);
	plane = rs->addQuad(vec3(0, 1, 0), vec3(0, 0, 0), 50.0f, 50.0f, planeMat);

	rs->addPointLight(vec3(5, 10, 2), vec3(100));
#endif
}

void App::loadInstances(rfw::utils::ArrayProxy<rfw::GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs)
{
#if CESIUMMAN
	cesiumManInstance = rs->addInstance(cesiumMan, vec3(1), vec3(10, 0.2f, 3));
#endif
#if POLLY
	pollyInstance = rs->addInstance(polly, vec3(1), vec3(8, 2, 1));
#endif
#if PICA
	picaInstance = rs->addInstance(pica);
	picaInstance.rotate(180.0f, vec3(0, 1, 0));
	picaInstance.update();
#endif
#if PICA_LIGHTS
	lightQuadInstance = rs->addInstance(lightQuad);
#endif
#if DRAGON
	dragonInstance = rs->addInstance(dragon, vec3(10), vec3(0, 2.83f, 0));
	planeInstance = rs->addInstance(plane);
#endif
}

void App::update(std::unique_ptr<rfw::RenderSystem> &rs, float dt)
{
	stats = rs->getRenderStats();
	if (playAnimations)
		rs->updateAnimationsTo(static_cast<float>(glfwGetTime()));

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
	{
		updateFocus = true;
		rs->setProbeIndex(uvec2(mouseX, mouseY));
		probe = rs->getProbeResult();
		instanceID = probe.object.getIndex();
		materialID = probe.materialIdx;

		hostMaterial = rs->getMaterial(materialID);
	}

	camera.aperture = max(camera.aperture, 0.00001f);
	camera.focalDistance = max(camera.focalDistance, 0.01f);

	if (updateFocus)
	{
		camera.focalDistance = max(probe.distance, 0.00001f);
		updateFocus = false;
		camChanged = true;
	}

	camera.aperture = max(camera.aperture, 0.00001f);
	camera.focalDistance = max(camera.focalDistance, 0.01f);

	if (camChanged)
	{
		status = Reset;
		camChanged = false;
	}
}

void App::renderGUI(std::unique_ptr<rfw::RenderSystem> &rs)
{
	ImGui::Begin("Stats");
	ImGui::Checkbox("Animations", &playAnimations);
	ImGui::Text("# Primary: %6ik (%2.1fM/s)", stats.primaryCount / 1000, stats.primaryCount / (max(1.0f, stats.primaryTime * 1000000)));
	ImGui::Text("# Secondary: %6ik (%2.1fM/s)", stats.secondaryCount / 1000, stats.secondaryCount / (max(1.0f, stats.secondaryTime * 1000000)));
	ImGui::Text("# Deep: %6ik (%2.1fM/s)", stats.deepCount / 1000, stats.deepCount / (max(1.0f, stats.deepTime * 1000000)));
	ImGui::Text("# Shadow: %6ik (%2.1fM/s)", stats.shadowCount / 1000, stats.shadowCount / (max(1.0f, stats.shadowTime * 1000000)));

	ImGui::Text("Primary %2.2f ms", stats.primaryTime);
	ImGui::Text("Secondary %2.2f ms", stats.secondaryTime);
	ImGui::Text("Deep %2.2f ms", stats.deepTime);
	ImGui::Text("Shadow %2.2f ms", stats.shadowTime);
	ImGui::Text("Shade %2.2f ms", stats.shadeTime);
	ImGui::Text("Finalize %2.2f ms", stats.finalizeTime);
	ImGui::Text("Animation %2.2f ms", stats.animationTime);
	ImGui::Text("Render %2.2f ms", stats.renderTime);

	ImGui::Separator();
	ImGui::BeginGroup();
	ImGui::Text("Camera");
	ImGui::SliderFloat("Contrast", &camera.contrast, 0.0f, 2.0f, "%.7f");
	ImGui::SliderFloat("Brightness", &camera.brightness, 0.0f, 2.0f, "%.7f");
	camChanged |= ImGui::DragFloat("Aperture", &camera.aperture, 0.0001f, 0.000001f, 1.0f, "%.7f");
	camChanged |= ImGui::DragFloat("Focal dist", &camera.focalDistance, 0.0001f, 0.00001f, 1e10f, "%.7f");
	camChanged |= ImGui::DragFloat("FOV", &camera.FOV, 0.1f, 20.0f, 120.0f);
	ImGui::DragFloat("Speed", &speedModifier, 0.1f, 0.001f, 10000.0f);
	ImGui::EndGroup();

	ImGui::Separator();

	ImGui::BeginGroup();
	ImGui::Text("Probe pixel (%u %u)", rs->getProbeIndex().x, rs->getProbeIndex().y);
	ImGui::Text("Distance %f", distance);
	ImGui::Text("Instance %zu", instanceID);
	try
	{
		auto instanceObject = rs->getInstanceReference(instanceID);

		auto instanceTranslation = instanceObject.getTranslation();
		auto instanceScaling = instanceObject.getScaling();
		auto instanceRotation = instanceObject.getRotation();

		ImGui::Text("Instance");
		if (ImGui::DragFloat3("Translation", value_ptr(instanceTranslation), 0.1f, -1e20f, 1e20f))
		{
			instanceObject.setTranslation(instanceTranslation);
			instanceObject.update();
		}
		if (ImGui::DragFloat3("Scaling", value_ptr(instanceScaling), 0.1f, -1e20f, 1e20f))
		{
			instanceObject.setScaling(instanceScaling);
			instanceObject.update();
		}
		if (ImGui::DragFloat3("Rotation", value_ptr(instanceRotation), 0.01f, -1e20f, 1e20f))
		{
			instanceObject.setRotation(instanceRotation);
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
		rs->setMaterial(materialID, hostMaterial);
	ImGui::Text("Material %zu", materialID);
	ImGui::ColorEdit3("Color", value_ptr(hostMaterial.color), ImGuiColorEditFlags_DisplayRGB);
	ImGui::SliderFloat3("Absorption", value_ptr(hostMaterial.absorption), 0.0f, 100.0f, "%.0f");
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

void App::cleanup() { camera.serialize("camera.bin"); }

int main(int argc, char *argv[])
{
	auto app = new App();
	rfw::Application::run(app);
}
