#include <Application.h>

#define SKINNED_MESH 0
#define PICA 1
#define PICA_LIGHTS 1
#define SPONZA 0
#define DRAGON 0
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
	rfw::GeometryReference cesiumMan;
	rfw::InstanceReference cesiumManInstance;
	rfw::GeometryReference pica;
	rfw::InstanceReference picaInstance;

	rfw::GeometryReference lightQuad;
	rfw::InstanceReference lightQuadInstance;
	rfw::LightReference pointLight;
	rfw::LightReference spotLight;

	rfw::RenderSystem::ProbeResult probe = {};
	size_t instanceID = 0, materialID = 0;
	rfw::HostMaterial hostMaterial = {};
	rfw::RenderStats stats = {};

	float distance = 0;
	float speedModifier = 0.2f;

	bool updateFocus = false;
	bool camChanged = false;
};

App::App() : Application(1280, 720, "RenderingFW", "VulkanRTX")
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
	cesiumMan = rs->addObject("Models/CesiumMan.glb", false, glm::scale(glm::mat4(1.0f), vec3(1.5)));
	pica = rs->addObject("Models/pica/scene.gltf");

	auto lightMaterial = rs->addMaterial(vec3(50), 1);

	lightQuad = rs->addQuad(vec3(0, -1, 0), vec3(0, 25, 0), 8.0f, 8.0f, lightMaterial);
	lightQuadInstance = rs->addInstance(lightQuad);
	pointLight = rs->addPointLight(vec3(-15, 10, -5), vec3(10));
	spotLight = rs->addSpotLight(vec3(10, 10, 3), cos(radians(30.0f)), vec3(10), cos(radians(45.0f)), vec3(0, -1, 0));
}

void App::loadInstances(rfw::utils::ArrayProxy<rfw::GeometryReference> geometry, std::unique_ptr<rfw::RenderSystem> &rs)
{
	cesiumManInstance = rs->addInstance(cesiumMan, vec3(1), vec3(10, 0.2f, 3));
	picaInstance = rs->addInstance(pica);

	picaInstance.rotate(180.0f, vec3(0, 1, 0));
	picaInstance.update();
}

void App::update(std::unique_ptr<rfw::RenderSystem> &rs, float dt)
{
	stats = rs->getRenderStats();

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

	ImGui::Separator();
	ImGui::BeginGroup();
	ImGui::Text("Camera");
	ImGui::DragFloat("Contrast", &camera.contrast, 0.0001f, 0.000001f, 2.0f, "%.7f");
	ImGui::DragFloat("Brightness", &camera.brightness, 0.0001f, 0.000001f, 10.0f, "%.7f");
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
	ImGui::ColorEdit3("Absorption", value_ptr(hostMaterial.absorption), ImGuiColorEditFlags_DisplayRGB);
	ImGui::DragFloat("Ior", &hostMaterial.eta, 0.01f, 0.0f, 5.0f);
	ImGui::DragFloat("Roughness", &hostMaterial.roughness, 0.05f, 0.0f, 5.0f);
	ImGui::DragFloat("Transmission", &hostMaterial.transmission, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Metallic", &hostMaterial.metallic, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("ClearCoat", &hostMaterial.clearcoat, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("ClearCoatGloss", &hostMaterial.clearcoatGloss, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Sheen", &hostMaterial.sheen, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Sheen tint", &hostMaterial.sheenTint, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Specular", &hostMaterial.specular, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Spec tint", &hostMaterial.specularTint, 0.01f, 0.0f, 1.0f);
	ImGui::DragFloat("Subsurface", &hostMaterial.subsurface, 0.01f, 0.0f, 1.0f);
	ImGui::EndGroup();

	ImGui::End();

	ImGui::End();
}

void App::cleanup() { camera.serialize("camera.bin"); }

int main(int argc, char *argv[])
{
	auto app = new App();
	rfw::Application::run(app);
}
