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

#define USE_GL_CONTEXT 0
#define CATCH_ERRORS 0

#define SKINNED_MESH 0
#define PICA 1
#define SPONZA 0

int main()
{
	using namespace rfw;
	using namespace utils;
	try
	{
#if USE_GL_CONTEXT
		auto window = std::make_shared<Window>(1600, 900, "Window", true, std::make_pair(4, 5));
#else
		// Create a window context with no initialized API
		auto window = std::make_shared<Window>(1600, 900, "Window", true);
#endif

		auto rs = RenderSystem();
		//		auto imguiContext = imgui::Context(window->getGLFW());

		unsigned int mouseX, mouseY;
		window->addMousePosCallback([&mouseX, &mouseY, &window](double x, double y, double lastX, double lastY) {
			mouseX = static_cast<uint>(x * double(window->getWidth()));
			mouseY = static_cast<uint>(y * double(window->getHeight()));
		});

		rs.loadRenderAPI("VkContext");
		rs.setTarget(window);

		auto camera = rfw::Camera();

		if (rfw::utils::file::exists("camera.bin"))
			camera = rfw::Camera::deserialize("camera.bin");

		camera.resize(window->getWidth(), window->getHeight());

		rs.setSkybox("Envmaps/sky_15.hdr");

#if SKINNED_MESH
		rs.addInstance(rs.addObject("Models/capture.DAE"), vec3(100), {0, vec3(1)});
#endif
#if PICA
		auto staticRef = rs.addObject("Models/pica/scene.gltf");
		auto staticInstanceRef = rs.addInstance(staticRef, vec3(1), vec3(0.0f), 180.0f, vec3(0, 1, 0));

		auto lightMaterial = rs.addMaterial(vec3(10), 1);
		auto lightQuad = rs.addQuad(vec3(0, -1, 0), vec3(-10, 25, 4), 10.0f, 10.0f, lightMaterial);
		auto lightInstance = rs.addInstance(lightQuad);

		rs.addPointLight(vec3(-15, 10, -5), 1.0f, vec3(1000));
		rs.addSpotLight(vec3(10, 10, 3), cos(radians(30.0f)), vec3(1000), cos(radians(45.0f)), 1.0f, vec3(0, -1, 0));
		rs.addDirectionalLight(vec3(0, -.8f, -1), 1.0f, vec3(1));
#endif

#if SPONZA
		camera.aperture = 0.001f;
		camera.focalDistance = 15.0f;
		auto lightMaterial = rendersystem.addMaterial(vec3(100), 1);
		auto lightQuad = rendersystem.addQuad(vec3(0, -1, 0), vec3(0, 600, 0), 80.0f, 200.0f, lightMaterial);
		auto lightInstance = rendersystem.addInstance(lightQuad);
		auto staticRef = rendersystem.addObject("Models/sponza/sponza.obj");
		auto staticInstanceRef = rendersystem.addInstance(staticRef, vec3(1), vec3(0.0f), 180.0f, vec3(0, 1, 0));
		auto legoCar = rendersystem.addObject("Models/legocar.obj");
		auto legoCarInstance = rendersystem.addInstance(legoCar, vec3(250), vec3(0, 27, 0), 90.0f, vec3(0, 1, 0));

		auto dragonMaterialIdx = rendersystem.addMaterial(vec3(1, 1, 1), 0.5f);
		auto dragonMaterial = rendersystem.getMaterial(dragonMaterialIdx);
		dragonMaterial.roughness = 0.6f;
		dragonMaterial.transmission = 0.95f;
		dragonMaterial.absorption = vec3(0.0f, 0.2f, 0.2f);
		dragonMaterial.color = vec3(1, 0, 0);
		dragonMaterial.eta = 1.4f;
		rendersystem.setMaterial(dragonMaterialIdx, dragonMaterial);
		auto dragon = rendersystem.addObject("Models/dragon.obj", static_cast<int>(dragonMaterialIdx));
		auto dragonInstance = rendersystem.addInstance(dragon, vec3(100), vec3(-120, 28, 0), 0.0f, vec3(1));
#endif
		window->addResizeCallback([&camera](int w, int h) {
			camera.resize(w, h);
			glViewport(0, 0, w, h);
		});

		std::future<void> prepareNextFrame, synchronize;

		Averager<float, 50> fpsStat;
		Averager<float, 50> synchronizeStat;
		Averager<float, 50> playAnimationStat;
		Averager<float, 50> renderStat;

		Timer timer;
		bool updateFocus = false;
		bool playAnimation = false;
		bool denoise = false;
		rs.synchronize();
		rs.setProbeIndex(glm::uvec2(window->getWidth() / 2, window->getHeight() / 2));
		while (!window->shouldClose())
		{
			bool camChanged = false;

			//			imguiContext.newFrame();
			RenderStatus status = window->pressed(KEY_B) ? Reset : Converge;
			const float elapsed = timer.elapsed();
			fpsStat.addSample(elapsed);

			//			ImGui::Begin("Stats");
			//			ImGui::Text("FPS: %3.1f", 1000.0f / fpsStat.getAverage());
			//			ImGui::Text("Frame time %3.1f ms", elapsed);
			//			ImGui::Text("Synchronize %3.1f ms", synchronizeStat.getAverage());
			//			ImGui::Text("Animation %3.1f ms", playAnimationStat.getAverage());
			//			ImGui::Text("Render %3.1f ms", renderStat.getAverage());
			//			ImGui::End();
			timer.reset();

			// Poll OS events
			window->pollEvents();

			auto translation = vec3(0.0f);
			auto target = vec3(0.0f);

			float speedModifier = 0.2f;

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
			if (window->pressed(Keycode::KEY_LEFT_SHIFT))
				speedModifier = 1.0f;
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

			translation *= elapsed * speedModifier * 0.01f;
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

			auto results = rs.getProbeResult();

//			ImGui::Begin("Settings");
//			ImGui::Checkbox("Autoplay", &playAnimation);
//			if (ImGui::Checkbox("Denoise", &denoise))
//				rs.setSetting({"denoise", denoise ? "1" : "0"});
//			camChanged |= ImGui::DragFloat("Aperture", &camera.aperture, 0.0001f, 0.000001f, 1.0f, "%.7f");
//			camChanged |= ImGui::DragFloat("Focal dist", &camera.focalDistance, 0.0001f, 0.00001f, 1e10f, "%.7f");
//			ImGui::End();

//			ImGui::Begin("Instance");
//			ImGui::Text("Instance %zu", (size_t)results.object);
//			auto instanceTranslation = results.object.getTranslation();
//			auto instanceScaling = results.object.getScaling();
//			if (ImGui::DragFloat3("Translation", value_ptr(instanceTranslation), 1.0f, -1e20f, 1e20f))
//			{
//				results.object.setTranslation(instanceTranslation);
//				results.object.update();
//			}
//			if (ImGui::DragFloat3("Scaling", value_ptr(instanceScaling), 1.0f, -1e20f, 1e20f))
//			{
//				results.object.setScaling(instanceScaling);
//				results.object.update();
//			}
//			ImGui::End();

//			ImGui::Begin("Prim Material");
//			auto hostMaterial = rs.getMaterial(results.materialIdx);
//			bool matChanged = false;
//			ImGui::Text("Material %zu", results.materialIdx);
//			matChanged |= ImGui::DragFloat3("Color", value_ptr(hostMaterial.color), 0.01f, 0.0f);
//			matChanged |= ImGui::DragFloat3("Absorption", value_ptr(hostMaterial.absorption), 0.01f, 0.0f);
//			matChanged |= ImGui::DragFloat("Ior", &hostMaterial.eta, 0.01f, 0.0f, 5.0f);
//			matChanged |= ImGui::DragFloat("Roughness", &hostMaterial.roughness, 0.05f, 0.0f, 5.0f);
//			matChanged |= ImGui::DragFloat("Transmission", &hostMaterial.transmission, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Metallic", &hostMaterial.metallic, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("ClearCoat", &hostMaterial.clearcoat, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("ClearCoatGloss", &hostMaterial.clearcoatGloss, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Sheen", &hostMaterial.sheen, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Sheen tint", &hostMaterial.sheenTint, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Specular", &hostMaterial.specular, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Spec tint", &hostMaterial.specularTint, 0.01f, 0.0f, 1.0f);
//			matChanged |= ImGui::DragFloat("Subsurface", &hostMaterial.subsurface, 0.01f, 0.0f, 1.0f);
//			ImGui::End();

//			if (matChanged)
//				rs.setMaterial(results.materialIdx, hostMaterial);

			if (camChanged)
				status = Reset;

			if (playAnimation)
			{
				Timer anim;
				rs.updateAnimationsTo(float(glfwGetTime()));
				playAnimationStat.addSample(anim.elapsed());
			}

			// Synchronize scene data
			Timer t;
			rs.synchronize();
			synchronizeStat.addSample(t.elapsed());

			//	Render scene
			t.reset();
			rs.renderFrame(camera, status);
			renderStat.addSample(t.elapsed());

//			imguiContext.render();
#if USE_GL_CONTEXT
			textureShader.bind();
			textureTarget->bind(0);
			drawQuad();
			textureShader.unbind();

			// Present result
			window->present();
#endif
		}

		DEBUG("Serializing camera.");
		camera.serialize("camera.bin");

		// Window callbacks must be cleaned up first since they might contain references
		window->clearCallbacks();
	}
	catch (const std::exception &e)
	{
		FAILURE("An exception occured: %s", e.what());
	}

	return 0;
}