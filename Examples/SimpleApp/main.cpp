#include <iostream>
#include <future>
#include <memory>

#include <RenderSystem.h>
#include <utils.h>
#include <utils/gl/GLTexture.h>
#include <utils/gl/GLShader.h>
#include <utils/gl/GLDraw.h>

#include <MaterialList.h>

#include <ImGuiContext.h>

#define USE_GL_CONTEXT 1
#define CATCH_ERRORS 0

#define SKINNED_MESH 0
#define PICA 1
#define PICA_LIGHTS 1
#define SPONZA 0
#define DRAGON 0
#define ANIMATE_DRAGON 0

int main(int argc, char *argv[])
{
	using namespace rfw;
	using namespace utils;

	auto rs = RenderSystem();

	auto camera = rfw::Camera::deserialize("camera.bin");

	unsigned int mouseX, mouseY;

	// try
	//{
#if USE_GL_CONTEXT
	auto window = std::make_shared<Window>(1280, 720, "Window", true, std::make_pair(4, 5));
	auto textureTarget = new GLTexture(GLTexture::VEC4, window->getFramebufferWidth(), window->getFramebufferHeight(), true);
	auto textureShader = GLShader("shaders/draw-tex.vert", "shaders/draw-tex.frag");

	textureShader.bind();
	textureTarget->bind(0);
	textureShader.setUniform("view", mat4(1.0f));
	textureShader.setUniform("tex", 0);
	textureShader.unbind();

	auto imguiContext = imgui::Context(window->getGLFW());

	window->addResizeCallback([&window, &textureTarget, &rs, &textureShader, &camera](int width, int height) {
		auto oldID = textureTarget->getID();
		textureTarget = new GLTexture(GLTexture::VEC4, window->getFramebufferWidth(), window->getFramebufferHeight(), true);
		rs.setTarget(textureTarget);
		textureShader.bind();
		textureTarget->bind(0);
		textureShader.setUniform("tex", 0);
		textureShader.unbind();
		glDeleteTextures(1, &oldID);

		camera.resize(width, height);
		glViewport(0, 0, width, height);
	});

	glViewport(0, 0, window->getWidth(), window->getHeight());
#else
	auto window = std::make_shared<Window>(1280, 720, "Window", true);

	window->addResizeCallback([&rs, &camera](int width, int height) { camera.resize(width, height); });
#endif

	camera.resize(window->getWidth(), window->getHeight());

	window->addMousePosCallback([&mouseX, &mouseY, &window](double x, double y, double lastX, double lastY) {
		mouseX = static_cast<uint>(x * double(window->getWidth()));
		mouseY = static_cast<uint>(y * double(window->getHeight()));
	});

	for (int i = 0; i < argc; i++)
	{
		const std::string arg = argv[i];
		if (arg == "-r" && argc >= (i + 1))
		{
			const std::string renderer = argv[i + 1];
			try
			{
				rs.loadRenderAPI(renderer);
				break;
			}
			catch (const std::exception &e)
			{
				WARNING("Attempted to load given renderer \"%s\" but error occured: %s", renderer.c_str(), e.what());
			}
		}
	}

	if (!rs.hasContext())
	{
		// Pick default renderer
		// rs.loadRenderAPI("GLRenderer"); // OpenGL PBR Renderer

		// rs.loadRenderAPI("OptiX6Context"); // OptiX 6.5 Path tracer

		rs.loadRenderAPI("VulkanRTX"); // Vulkan RTX Path tracer

		// rs.loadRenderAPI("VkContext"); // WIP Vulkan PBR Renderer
	}

	rs.setSkybox("Envmaps/sky_15.hdr");

#if USE_GL_CONTEXT
	rs.setTarget(textureTarget);
#else
	rs.setTarget(window);
#endif

#if SKINNED_MESH
	auto skinnedMesh = rs.addInstance(rs.addObject("Models/capture.DAE"), vec3(10));
#endif

	auto directionalLight = rs.addDirectionalLight(vec3(0, -.8f, -1), vec3(1));
	auto cesiumMan = rs.addInstance(rs.addObject("Models/CesiumMan.glb", false, glm::scale(glm::mat4(1.0f), vec3(1.5))), vec3(1), vec3(10, 0.2f, 3));

	auto projectPolly = rs.addInstance(rs.addObject("Models/project_polly.glb"), vec3(2), vec3(0, 5, 0), 90.0f, vec3(0, 1, 0));
	// auto interpolationTest = rs.addInstance(rs.addObject("Models/InterpolationTest.glb"), vec3(-1, 1, -1), vec3(0, 10, 0));
	auto animatedCube = rs.addInstance(rs.addObject("Models/AnimatedMorphCube.glb"), vec3(1, -1, -1), vec3(-5, 2, 0), 90.0f, vec3(1, 0, 0));
	auto animatedSphere = rs.addInstance(rs.addObject("Models/AnimatedMorphSphere.glb"), vec3(1), vec3(5, 2, -4), 90.0f, vec3(1, 0, 0));

#if PICA
	auto lightMaterial = rs.addMaterial(vec3(100), 1);
	auto staticRef = rs.addObject("Models/pica/scene.gltf");
	auto staticInstanceRef = rs.addInstance(staticRef);
	staticInstanceRef.rotate(180.0f, vec3(0, 1, 0));
	staticInstanceRef.update();

	auto lightQuad = rs.addQuad(vec3(0, -1, 0), vec3(0, 25, 0), 8.0f, 8.0f, lightMaterial);
	auto lightInstance = rs.addInstance(lightQuad);
#if PICA_LIGHTS
	auto pointLight = rs.addPointLight(vec3(-15, 10, -5), vec3(100));
	auto spotLight = rs.addSpotLight(vec3(10, 10, 3), cos(radians(30.0f)), vec3(100), cos(radians(45.0f)), vec3(0, -1, 0));
#endif
#endif

#if SPONZA
	camera.aperture = 0.001f;
	camera.focalDistance = 15.0f;
	auto lightMaterial = rs.addMaterial(vec3(30), 1);
	auto lightQuad = rs.addQuad(vec3(0, -1, 0), vec3(0, 10, 0), 30.0f, 120.0f, lightMaterial);
	auto lightInstance = rs.addInstance(lightQuad);
	auto staticRef = rs.addObject("Models/sponza/sponza.obj", false);
	auto staticInstanceRef = rs.addInstance(staticRef, vec3(0.1f), vec3(0.0f), 180.0f, vec3(0, 1, 0));
#endif

#if DRAGON
	auto dragonMaterialIdx = rs.addMaterial(vec3(1, 1, 1), 0.5f);
	auto dragonMaterial = rs.getMaterial(dragonMaterialIdx);
	dragonMaterial.roughness = 0.6f;
	dragonMaterial.transmission = 0.95f;
	dragonMaterial.absorption = vec3(0.0f, 0.2f, 0.2f);
	dragonMaterial.color = vec3(1, 0, 0);
	dragonMaterial.eta = 1.4f;
	rs.setMaterial(dragonMaterialIdx, dragonMaterial);
	auto dragon = rs.addObject("Models/dragon.obj", static_cast<int>(dragonMaterialIdx));
	auto dragonInstance = rs.addInstance(dragon, vec3(10), vec3(-20, 4.0f, 0), 0.0f, vec3(1));
#endif

	// auto dragonMaterialIdx = rs.addMaterial(vec3(1, 1, 1), 0.5f);
	// auto dragonMaterial = rs.getMaterial(dragonMaterialIdx);
	// dragonMaterial.roughness = 0.6f;
	// dragonMaterial.transmission = 0.95f;
	// dragonMaterial.absorption = vec3(0.0f, 0.2f, 0.2f);
	// dragonMaterial.color = vec3(1, 0, 0);
	// dragonMaterial.eta = 1.4f;
	// rs.setMaterial(dragonMaterialIdx, dragonMaterial);
	// auto dragon = rs.addObject("Models/dragon.obj", static_cast<int>(dragonMaterialIdx));

	// for (int x = -20; x <= 20; x++)
	//{
	//	for (int z = -20; z <= 20; z++)
	//	{
	//		auto dragonInstance = rs.addInstance(dragon, vec3(10), vec3(10 * x, 0.0f, 10 * z));
	//	}
	//}

	std::future<void> prepareNextFrame, synchronize;

	Averager<float, 50> fpsStat;
	Averager<float, 50> synchronizeStat;
	Averager<float, 50> playAnimationStat;
	Averager<float, 50> renderStat;

	Averager<float, 50> primaryStat;
	Averager<float, 50> secondaryStat;
	Averager<float, 50> deepStat;
	Averager<float, 50> shadowStat;
	Averager<float, 50> shadeStat;
	Averager<float, 50> finalizeStat;

	const auto bounds = rs.calculateSceneBounds();

	DEBUG("Scene bounds: min(%f, %f, %f), max(%f, %f, %f)", bounds.mMin.x, bounds.mMin.y, bounds.mMin.z, bounds.mMax.x, bounds.mMax.y, bounds.mMax.z);

	Timer timer;
	bool updateFocus = false;
	bool playAnimation = false;
	bool denoise = false;

	rfw::HostMaterial hostMaterial = {};
	rfw::RenderStats stats = {};
	float distance = 0;
	float speedModifier = 0.2f;

	size_t instanceIdx = 0;
	size_t materialIdx = 0;

#if USE_GL_CONTEXT
	// Preparse renderer settings to be used with ImGui
	const auto settings = rs.getAvailableSettings();
	std::vector<int> settingIndices(settings.settingKeys.size(), 0);
	std::vector<const char *> keys(settings.settingKeys.size());
	std::vector<std::vector<const char *>> values(settings.settingKeys.size());
	for (int i = 0, s = static_cast<int>(settings.settingKeys.size()); i < s; i++)
	{
		keys[i] = settings.settingKeys[i].c_str();
		values[i].resize(settings.settingValues[i].size());
		for (int j = 0, sj = static_cast<int>(settings.settingValues[i].size()); j < sj; j++)
			values[i][j] = settings.settingValues[i][j].c_str();
	}
#endif

	rs.synchronize();
	rs.setProbeIndex(glm::uvec2(window->getWidth() / 2, window->getHeight() / 2));
	while (!window->shouldClose())
	{
		bool camChanged = false;

#if USE_GL_CONTEXT
		imguiContext.newFrame();
#endif
		RenderStatus status = window->pressed(KEY_B) ? Reset : Converge;
		const float elapsed = timer.elapsed();
		fpsStat.addSample(elapsed);
		timer.reset();

		// Poll OS events
		window->pollEvents();

		auto translation = vec3(0.0f);
		auto target = vec3(0.0f);

		if (window->pressed(KEY_EQUAL))
		{
			camChanged = true;
			camera.aperture += 0.0001f;
		}
		if (window->pressed(KEY_MINUS))
		{
			camChanged = true;
			camera.aperture -= 0.0001f;
		}

		if (window->pressed(KEY_LEFT_BRACKET))
		{
			camChanged = true;
			camera.focalDistance -= 0.1f;
		}
		if (window->pressed(KEY_RIGHT_BRACKET))
		{
			camChanged = true;
			camera.focalDistance += 0.1f;
		}

		camera.aperture = max(camera.aperture, 0.00001f);
		camera.focalDistance = max(camera.focalDistance, 0.01f);

		// Key handling
		if (window->pressed(Keycode::KEY_ESCAPE))
			window->close();
		if (window->pressed(Keycode::KEY_W))
			translation.z += 1.0f;
		if (window->pressed(Keycode::KEY_S))
			translation.z -= 1.0f;
		if (window->pressed(Keycode::KEY_D))
			translation.x += 1.0f;
		if (window->pressed(Keycode::KEY_A))
			translation.x -= 1.0f;
		if (window->pressed(Keycode::KEY_R))
			translation.y += 1.0f;
		if (window->pressed(Keycode::KEY_F))
			translation.y -= 1.0f;
		if (window->pressed(Keycode::KEY_UP))
			target.y += 0.001f;
		if (window->pressed(Keycode::KEY_DOWN))
			target.y -= 0.001f;
		if (window->pressed(Keycode::KEY_RIGHT))
			target.x += 0.001f;
		if (window->pressed(Keycode::KEY_LEFT))
			target.x -= 0.001f;

		translation *= elapsed * speedModifier * 0.01f * (window->pressed(Keycode::KEY_LEFT_SHIFT) ? 5.0f : 1.0f);
		target *= elapsed;

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

		if (updateFocus)
		{
			const auto results = rs.getProbeResult();
			camera.focalDistance = max(results.distance, 0.00001f);
			updateFocus = false;
			camChanged = true;
		}

		if (window->mousePressed(Mousekey::BUTTON_RIGHT))
		{
			updateFocus = true;
			rs.setProbeIndex(uvec2(mouseX, mouseY));
		}

		stats = rs.getRenderStats();
		primaryStat.addSample(stats.primaryTime);
		secondaryStat.addSample(stats.secondaryTime);
		deepStat.addSample(stats.deepTime);
		shadowStat.addSample(stats.shadowTime);
		shadeStat.addSample(stats.shadeTime);
		finalizeStat.addSample(stats.finalizeTime);

#if USE_GL_CONTEXT
		ImGui::Begin("Settings");
		ImGui::BeginGroup();
		ImGui::Text("Statistics");
		ImGui::Text("FPS: %3.1f", 1000.0f / fpsStat.getAverage());
		const auto frameTimeData = fpsStat.data();
		ImGui::PlotLines("Frame Times", frameTimeData.data(), static_cast<int>(frameTimeData.size()));

		ImGui::Separator();

		ImGui::Text("Primary %2.2f ms", primaryStat.getAverage());
		ImGui::Text("Secondary %2.2f ms", secondaryStat.getAverage());
		ImGui::Text("Deep %2.2f ms", deepStat.getAverage());
		ImGui::Text("Shadow %2.2f ms", shadowStat.getAverage());
		ImGui::Text("Shade %2.2f ms", shadeStat.getAverage());
		ImGui::Text("Finalize %2.2f ms", finalizeStat.getAverage());

		ImGui::Separator();

		ImGui::Text("# Primary: %6ik (%2.1fM/s)", stats.primaryCount / 1000, stats.primaryCount / (max(1.0f, primaryStat.getAverage() * 1000000)));
		ImGui::Text("# Secondary: %6ik (%2.1fM/s)", stats.secondaryCount / 1000, stats.secondaryCount / (max(1.0f, secondaryStat.getAverage() * 1000000)));
		ImGui::Text("# Deep: %6ik (%2.1fM/s)", stats.deepCount / 1000, stats.deepCount / (max(1.0f, deepStat.getAverage() * 1000000)));
		ImGui::Text("# Shadow: %6ik (%2.1fM/s)", stats.shadowCount / 1000, stats.shadowCount / (max(1.0f, shadowStat.getAverage() * 1000000)));

		ImGui::Separator();

		ImGui::Text("Synchronize %3.1f ms", synchronizeStat.getAverage());
		ImGui::Text("Animation %3.1f ms", playAnimation ? playAnimationStat.getAverage() : 0.0f);
		ImGui::Text("Render %3.1f ms", renderStat.getAverage());
		ImGui::EndGroup();

		ImGui::Separator();

		ImGui::BeginGroup();
		ImGui::Text("Context");
		ImGui::Checkbox("Autoplay", &playAnimation);
		for (int i = 0, s = static_cast<int>(keys.size()); i < s; i++)
		{
			if (ImGui::ListBox(keys[i], &settingIndices[i], values[i].data(), static_cast<int>(values[i].size())))
				rs.setSetting(rfw::RenderSetting(keys[i], values[i][settingIndices[i]]));
		}
		ImGui::Separator();

		camChanged |= ImGui::DragFloat("Aperture", &camera.aperture, 0.0001f, 0.000001f, 1.0f, "%.7f");
		camChanged |= ImGui::DragFloat("Focal dist", &camera.focalDistance, 0.0001f, 0.00001f, 1e10f, "%.7f");
		camChanged |= ImGui::DragFloat("FOV", &camera.FOV, 0.1f, 20.0f, 120.0f);
		ImGui::DragFloat("Speed", &speedModifier, 0.1f, 0.001f, 10000.0f);
		ImGui::Text("Cam Dir: %.2f, %.2f, %.2f", camera.direction.x, camera.direction.y, camera.direction.z);
		ImGui::EndGroup();

		ImGui::Separator();
		ImGui::BeginGroup();
		ImGui::Text("Probe pixel (%u %u)", rs.getProbeIndex().x, rs.getProbeIndex().y);

		if (ImGui::Button("Probe") || window->mousePressed(Mousekey::BUTTON_RIGHT))
		{
			const auto results = rs.getProbeResult();
			distance = results.distance;
			instanceIdx = results.object.getIndex();
			materialIdx = results.materialIdx;

			hostMaterial = rs.getMaterial(materialIdx);
		}

		ImGui::Text("Distance %f", distance);
		ImGui::Text("Instance %zu", instanceIdx);
		try
		{
			auto instanceObject = rs.getInstanceReference(instanceIdx);

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
			rs.setMaterial(materialIdx, hostMaterial);
		ImGui::Text("Material %zu", materialIdx);
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

		if (camChanged)
			status = Reset;

		if (playAnimation)
		{
			const auto time = float(getElapsedMicroSeconds() * (1.0 / 10e5));

			Timer anim;
			rs.updateAnimationsTo(time);
			playAnimationStat.addSample(anim.elapsed());
		}
#endif

		// Synchronize scene data
		Timer t;
		rs.synchronize();
		synchronizeStat.addSample(t.elapsed());

#if ANIMATE_DRAGON
		const float time = float(glfwGetTime());
		dragonInstance.setRotation(time * 10, vec3(.2f, -.2f, .1f));
		dragonInstance.update();
#endif

		//	Render scene
		t.reset();
		rs.renderFrame(camera, status);
		renderStat.addSample(t.elapsed());

#if USE_GL_CONTEXT
		glViewport(0, 0, window->getFramebufferWidth(), window->getFramebufferHeight());
		textureShader.bind();
		textureTarget->bind(0);
		textureShader.setUniform("tex", 0);
		drawQuad();
		textureShader.unbind();
		imguiContext.render();

		// Present result
		window->present();
#endif
	}

	DEBUG("Serializing camera.");
	camera.serialize("camera.bin");

	// Window callbacks must be cleaned up first since they might contain references
	window->clearCallbacks();
	//}
	// catch (const std::exception &e)
	//{
	//	std::cout << "Exception occurred: " << e.what() << std::endl;
	//	std::cin.get();
	//}

	return 0;
}