#include "rfw.h"

void rfw::Application::run(Application *app)
{
	app->init(app->m_RS);
	app->load_instances(app->m_RS->get_geometry(), app->m_RS);

	auto &rs = app->m_RS;

	utils::Timer t;
	while (!app->window.shouldClose())
	{
		app->window.pollEvents();

		rs->synchronize();
		rs->render_frame(app->camera, app->status, true);
		app->m_ImGuiContext.newFrame();
		app->post_render(rs);
		app->draw();
		const float elapsed = t.elapsed();
		t.reset();
		app->update(rs, elapsed);
	}

	app->cleanup();
}

rfw::Application::Application(size_t scrWidth, size_t scrHeight, std::string title, std::string renderAPI, bool hidpi)
	: window(static_cast<int>(scrWidth), static_cast<int>(scrHeight), title.data(), true, hidpi, std::make_pair(4, 5)), m_ImGuiContext(window.getGLFW())
{
	m_RS = std::make_unique<rfw::RenderSystem>();

	m_Target = new rfw::utils::GLTexture(rfw::utils::GLTexture::VEC4, window.get_render_width(), window.get_render_height(), true);
	m_Shader = new rfw::utils::GLShader("shaders/draw-tex-fxaa.vert", "shaders/draw-tex-fxaa.frag");

	m_Shader->bind();
	m_Target->bind(0);
	m_Shader->setUniform("view", mat4(1.0f));
	m_Shader->setUniform("tex", 0);
	m_Shader->unbind();

	window.addResizeCallback([this](int width, int height) {
		auto oldID = m_Target->getID();
		m_Target = new rfw::utils::GLTexture(rfw::utils::GLTexture::VEC4, window.get_render_width(), window.get_render_height(), true);
		m_RS->set_target(m_Target);
		m_Shader->bind();
		m_Target->bind(0);
		m_Shader->setUniform("tex", 0);
		m_Shader->setUniform("rt_w", 1.0f / float(window.get_render_width()));
		m_Shader->setUniform("rt_h", 1.0f / float(window.get_render_height()));
		m_Shader->unbind();
		glDeleteTextures(1, &oldID);

		camera.resize(width, height);
		glViewport(0, 0, width, height);
	});

	try
	{
		DEBUG("Loading render API: %s", renderAPI.c_str());
		m_RS->load_render_api(renderAPI);
		m_RS->set_target(m_Target);
		camera.resize(int(scrWidth), int(scrHeight));
	}
	catch (const std::exception &e)
	{
		FAILURE("Attempted to load given renderer %s but error occured: %s", renderAPI.c_str(), e.what());
	}

	// Preparse renderer settings to be used with ImGui
	renderSettings = m_RS->get_available_settings();
	settingKeys.resize(renderSettings.settingKeys.size());
	settingsCurrentValues.resize(renderSettings.settingKeys.size(), 0);
	settingAvailableValues.resize(settingKeys.size());
	for (int i = 0, s = static_cast<int>(renderSettings.settingKeys.size()); i < s; i++)
	{
		settingKeys[i] = renderSettings.settingKeys[i].c_str();
		settingAvailableValues[i].resize(renderSettings.settingValues[i].size());
		for (int j = 0, sj = static_cast<int>(renderSettings.settingValues[i].size()); j < sj; j++)
			settingAvailableValues[i][j] = renderSettings.settingValues[i][j].c_str();
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
	glViewport(0, 0, window.get_render_width(), window.get_render_height());
}
