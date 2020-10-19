#include "rfw.h"

void rfw::app::run(app &application)
{
	application.init(application.m_RS);
	application.load_instances(application.m_RS->get_geometry(), application.m_RS);

	auto &rs = application.m_RS;

	utils::timer t;
	while (!application.window.should_close())
	{
		application.window.poll_events();

		rs->synchronize();
		rs->render_frame(application.camera, application.status, true);
		application.m_ImGuiContext.newFrame();
		application.post_render(rs);
		application.draw();
		const float elapsed = t.elapsed();
		t.reset();
		application.update(rs, elapsed);
	}

	application.cleanup();
}

rfw::app::app(size_t scrWidth, size_t scrHeight, std::string title, std::string renderAPI, bool hidpi)
	: window(static_cast<int>(scrWidth), static_cast<int>(scrHeight), title.data(), true, hidpi, std::make_pair(4, 5)),
	  m_ImGuiContext(window.getGLFW())
{
	m_RS = std::make_unique<rfw::system>();

	m_Target =
		new rfw::utils::texture(rfw::utils::texture::VEC4, window.get_render_width(), window.get_render_height(), true);
	m_Shader = new rfw::utils::shader("shaders/draw-tex-fxaa.vert", "shaders/draw-tex-fxaa.frag");

	m_Shader->bind();
	m_Target->bind(0);
	m_Shader->set_uniform("view", mat4(1.0f));
	m_Shader->set_uniform("tex", 0);
	m_Shader->unbind();

	window.add_resize_callback([this](int width, int height) {
		auto oldID = m_Target->getID();
		m_Target = new rfw::utils::texture(rfw::utils::texture::VEC4, window.get_render_width(),
										   window.get_render_height(), true);
		m_RS->set_target(m_Target);
		m_Shader->bind();
		m_Target->bind(0);
		m_Shader->set_uniform("tex", 0);
		m_Shader->set_uniform("rt_w", 1.0f / float(window.get_render_width()));
		m_Shader->set_uniform("rt_h", 1.0f / float(window.get_render_height()));
		m_Shader->unbind();
		glDeleteTextures(1, &oldID);

		camera.resize(width, height);
		glViewport(0, 0, width, height);
	});

	try
	{
		DEBUG("Loading render API: %s", renderAPI.c_str());
		m_RS->load_render_api(renderAPI);
	}
	catch (const std::exception &e)
	{
		WARNING("Attempted to load given renderer %s but error occured: %s", renderAPI.c_str(), e.what());
		DEBUG("Loading render API: %s", GLRENDERER);
		m_RS->load_render_api(GLRENDERER);
	}

	m_RS->set_target(m_Target);
	camera.resize(int(scrWidth), int(scrHeight));

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

rfw::app::~app()
{
	m_RS.reset();

	delete m_Target;
	delete m_Shader;
}

void rfw::app::draw()
{
	glViewport(0, 0, window.get_framebuffer_width(), window.get_framebuffer_height());
	m_Shader->bind();
	m_Target->bind(0);
	m_Shader->set_uniform("tex", 0);
	utils::draw_quad();
	m_Shader->unbind();
	m_ImGuiContext.render();

	window.present();
	glViewport(0, 0, window.get_render_width(), window.get_render_height());
}
