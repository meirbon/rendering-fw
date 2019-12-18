#ifdef _WIN32
#include <Windows.h>
#endif

#include "Application.h"
#include "utils/gl/GLDraw.h"
#include "utils/Timer.h"

void rfw::Application::run(Application *app)
{
	app->init();
	app->loadScene(app->m_RS);
	app->loadInstances(app->m_RS->getGeometry(), app->m_RS);

	auto &rs = app->m_RS;

	utils::Timer t;
	while (!app->window.shouldClose())
	{
		app->window.pollEvents();

		rs->synchronize();
		rs->renderFrame(app->camera, app->status, true);
		app->m_ImGuiContext.newFrame();
		app->renderGUI(rs);
		app->draw();
		const float elapsed = t.elapsed();
		t.reset();
		app->update(rs, elapsed);
	}

	app->cleanup();
}

rfw::Application::Application(size_t scrWidth, size_t scrHeight, std::string_view title, std::string_view renderAPI)
	: window(static_cast<int>(scrWidth), static_cast<int>(scrHeight), title.data(), true, std::make_pair(4, 5)), m_ImGuiContext(window.getGLFW())
{
	m_RS = std::make_unique<rfw::RenderSystem>();

	m_Target = new rfw::utils::GLTexture(rfw::utils::GLTexture::VEC4, window.getFramebufferWidth(), window.getFramebufferHeight(), true);
	m_Shader = new rfw::utils::GLShader("shaders/draw-tex.vert", "shaders/draw-tex.frag");

	m_Shader->bind();
	m_Target->bind(0);
	m_Shader->setUniform("view", mat4(1.0f));
	m_Shader->setUniform("tex", 0);
	m_Shader->unbind();

	window.addResizeCallback([this](int width, int height) {
		auto oldID = m_Target->getID();
		m_Target = new rfw::utils::GLTexture(rfw::utils::GLTexture::VEC4, window.getFramebufferWidth(), window.getFramebufferHeight(), true);
		m_RS->setTarget(m_Target);
		m_Shader->bind();
		m_Target->bind(0);
		m_Shader->setUniform("tex", 0);
		m_Shader->unbind();
		glDeleteTextures(1, &oldID);

		camera.resize(width, height);
		glViewport(0, 0, width, height);
	});

	try
	{
		m_RS->loadRenderAPI(renderAPI);
		m_RS->setTarget(m_Target);
		camera.resize(scrWidth, scrHeight);
	}
	catch (const std::exception &e)
	{
		FAILURE("Attempted to load given renderer \"%s\" but error occured: %s", renderAPI.data(), e.what());
	}
}

rfw::Application::~Application()
{
	m_RS.reset();

	delete m_Target;
	delete m_Shader;
}

void rfw::Application::draw()
{
	glViewport(0, 0, window.getFramebufferWidth(), window.getFramebufferHeight());
	m_Shader->bind();
	m_Target->bind(0);
	m_Shader->setUniform("tex", 0);
	utils::drawQuad();
	m_Shader->unbind();
	m_ImGuiContext.render();

	window.present();
}
