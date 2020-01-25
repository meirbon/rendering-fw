#include "PCH.h"

using namespace rfw;
using namespace utils;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context()
{
	for (GLMesh *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();
	m_Textures.clear();

	delete m_SimpleShader;
	delete m_ColorShader;
	delete m_NormalShader;
}

std::vector<RenderTarget> Context::get_supported_targets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		auto error = glewInit();
		if (error != GLEW_NO_ERROR)
			throw std::runtime_error("Could not init GLEW.");
		m_InitializedGlew = true;

		m_SimpleShader = new GLShader("glshaders/simple.vert", "glshaders/simple.frag");
		m_ColorShader = new GLShader("glshaders/simple.vert", "glshaders/color.frag");
		m_NormalShader = new GLShader("glshaders/simple.vert", "glshaders/normal.frag");
		m_CurrentShader = m_SimpleShader;
		CheckGL();
	}

	if (m_FboID)
	{
		glDeleteFramebuffers(1, &m_FboID);
		glDeleteRenderbuffers(1, &m_RboID);

		m_FboID = 0;
		m_RboID = 0;
	}

	m_Width = width;
	m_Height = height;

	m_TargetID = *glTextureID;
	glGenFramebuffers(1, &m_FboID);
	glGenRenderbuffers(1, &m_RboID);

	glBindRenderbuffer(GL_RENDERBUFFER, m_RboID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_Width, m_Height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_RboID);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	assert(m_TargetID);
	assert(m_FboID);
	assert(m_RboID);
	CheckGL();
}

void Context::cleanup()
{
	if (m_FboID)
	{
		glDeleteFramebuffers(1, &m_FboID), m_FboID = 0;
		glDeleteRenderbuffers(1, &m_RboID), m_RboID = 0;
	}

	for (auto *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();

	m_Textures.clear();
	m_Materials.clear();
}

void Context::render_frame(const Camera &camera, RenderStatus status)
{
	m_RenderStats.clear();

	for (int i = 0, s = static_cast<int>(m_Textures.size()); i < s; i++)
		m_Textures[i].bind(i);

	auto timer = utils::Timer();
	CheckGL();
	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0, 0, 0, 1.0f);

	glViewport(0, 0, m_Width, m_Height);

	m_CurrentShader->bind();
	auto matrix = camera.get_matrix(0.01f, 1e16f);
	const auto matrix3x3 = mat3(matrix);
	m_CurrentShader->setUniform("CamMatrix", matrix);
	m_CurrentShader->setUniform("CamMatrix3x3", matrix3x3);
	m_CurrentShader->setUniform("CamDir", camera.direction);
	m_CurrentShader->setUniform("ambient", m_Ambient);
	m_CurrentShader->setUniform("forward", -camera.direction);

	for (int i = 0; i < m_Textures.size(); i++)
	{
		m_Textures.at(i).bind(i);

		char buffer[128];
		sprintf(buffer, "textures[%i]", i);
		m_CurrentShader->setUniform(buffer, m_TextureBindings.at(i));
	}

	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		const GLMesh *mesh = m_Meshes.at(i);
		const std::vector<glm::mat4> &instance = m_Instances[i];
		const std::vector<glm::mat4> &inverseInstance = m_InverseInstances[i];

		if (instance.empty())
			continue;

		mesh->vao.bind();

		int sj = 0;
		if (instance.size() < 32)
			sj = 1;
		else
			sj = static_cast<int>(instance.size() / 32);

		for (int j = 0; j < sj; j++)
		{
			const auto offset = j * 32;
			const auto count = min(32, int(instance.size()) - offset);

			// Update instance matrices
			m_CurrentShader->setUniform("InstanceMatrices[0]", instance.data() + offset, count, false);
			m_CurrentShader->setUniform("InverseMatrices[0]", inverseInstance.data() + offset, count, false);
			mesh->draw(*m_CurrentShader, count, m_Materials.data(), m_Textures.data());
		}
	}

	m_CurrentShader->unbind();

	m_RenderStats.primaryCount = m_Width * m_Height;
	m_RenderStats.primaryTime = timer.elapsed();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);
}

void Context::set_materials(const std::vector<DeviceMaterial> &materials, const std::vector<MaterialTexIds> &texDescriptors) { m_Materials = materials; }

void Context::set_textures(const std::vector<TextureData> &textures)
{
	CheckGL();

	GLint value;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &value);

	if (value < textures.size())
		FAILURE("Too many textures supplied, maximum supported by current GPU: %i", value);

	for (auto &m_Texture : m_Textures)
		m_Texture.cleanup();
	m_Textures.clear();

	m_Textures.resize(textures.size());
	m_TextureBindings.resize(textures.size());

	CheckGL();

	for (int i = 0, s = static_cast<int>(m_Textures.size()); i < s; i++)
	{
		const auto &tex = textures[i];
		auto &glTex = m_Textures[i];

		glTex.bind();
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		CheckGL();

		if (tex.type == TextureData::FLOAT4)
			glTex.setData(static_cast<vec4 *>(tex.data), tex.width, tex.height);
		else
			glTex.setData(static_cast<uint *>(tex.data), tex.width, tex.height);

		glTex.generateMipMaps();
		glTex.bind(i);

		m_TextureBindings[i] = i;
	}

	CheckGL();
}

void Context::set_mesh(size_t index, const rfw::Mesh &mesh)
{
	if (m_Meshes.size() <= index)
	{
		while (m_Meshes.size() <= index)
			m_Meshes.push_back(new GLMesh());
	}
	CheckGL();
	m_Meshes.at(index)->setMesh(mesh);
	CheckGL();
}

void Context::set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
{
	if (m_InstanceGeometry.size() <= i)
	{
		m_InstanceGeometry.push_back(int(meshIdx));
		m_InstanceMatrices.push_back(transform);
		m_InverseInstanceMatrices.push_back(inverse_transform);
	}
	else
	{
		m_InstanceGeometry[i] = int(meshIdx);
		m_InstanceMatrices[i] = transform;
		m_InverseInstanceMatrices[i] = inverse_transform;
	}
}

void Context::set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	CheckGL();
	std::vector<glm::vec4> skybox(width * height);
	for (int i = 0, s = static_cast<int>(width * height); i < s; i++)
		skybox[i] = vec4(pixels[i], 1);

	m_Skybox.setData(skybox, width, height);
	CheckGL();
}

void Context::set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
						 const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights)
{
	m_LightCount = lightCount;

	m_AreaLights.resize(lightCount.areaLightCount);
	if (!m_AreaLights.empty())
		memcpy(m_AreaLights.data(), areaLights, m_AreaLights.size() * sizeof(AreaLight));

	m_PointLights.resize(lightCount.pointLightCount);
	if (!m_PointLights.empty())
		memcpy(m_PointLights.data(), pointLights, m_PointLights.size() * sizeof(PointLight));

	m_DirectionalLights.resize(lightCount.directionalLightCount);
	if (!m_DirectionalLights.empty())
		memcpy(m_DirectionalLights.data(), directionalLights, m_DirectionalLights.size() * sizeof(DirectionalLight));

	m_SpotLights.resize(lightCount.spotLightCount);
	if (!m_SpotLights.empty())
		memcpy(m_SpotLights.data(), spotLights, m_SpotLights.size() * sizeof(SpotLight));

	setLights(m_SimpleShader);
}

void Context::get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::get_settings() const
{
	auto settings = rfw::AvailableRenderSettings();

	settings.settingKeys = {"VIEW"};
	settings.settingValues = {{"SHADED", "NORMAL", "COLOR"}};

	return settings;
}

void Context::set_setting(const rfw::RenderSetting &setting)
{
	if (setting.name == "VIEW")
	{
		if (setting.value == "SHADED")
			m_CurrentShader = m_SimpleShader;
		else if (setting.value == "NORMAL")
			m_CurrentShader = m_NormalShader;
		else if (setting.value == "COLOR")
			m_CurrentShader = m_ColorShader;
	}
}

void Context::update()
{
	m_Instances.resize(m_Meshes.size());
	m_InverseInstances.resize(m_Meshes.size());
	for (auto &i : m_Instances)
		i.clear();
	for (auto &i : m_InverseInstances)
		i.clear();

	for (int i = 0, s = static_cast<int>(m_InstanceGeometry.size()); i < s; i++)
	{
		const auto geoIdx = m_InstanceGeometry[i];

		const auto &matrix = m_InstanceMatrices[i];
		const auto &inverseMatrix = m_InverseInstanceMatrices[i];

		m_Instances[geoIdx].push_back(matrix);
		m_InverseInstances[geoIdx].push_back(inverseMatrix);
	}
}

void Context::set_probe_index(glm::uvec2 probePos) {}

rfw::RenderStats Context::get_stats() const { return m_RenderStats; }

void Context::setLights(utils::GLShader *shader)
{
	shader->bind();
	shader->setUniform("lightCount",
					   uvec4(m_LightCount.areaLightCount, m_LightCount.pointLightCount, m_LightCount.spotLightCount, m_LightCount.directionalLightCount));

	for (int i = 0, s = static_cast<int>(m_AreaLights.size()); i < s; i++)
	{
		const auto &l = m_AreaLights.at(i);

		char buffer[128];
		sprintf(buffer, "areaLights[%i].position_area", i);
		shader->setUniform(buffer, vec4(l.position, Triangle::calculateArea(l.vertex0, l.vertex1, l.vertex2)));
		sprintf(buffer, "areaLights[%i].normal", i);
		shader->setUniform(buffer, l.normal);
		sprintf(buffer, "areaLights[%i].radiance", i);
		shader->setUniform(buffer, l.radiance);
		sprintf(buffer, "areaLights[%i].vertex0", i);
		shader->setUniform(buffer, l.vertex0);
		sprintf(buffer, "areaLights[%i].vertex1", i);
		shader->setUniform(buffer, l.vertex1);
		sprintf(buffer, "areaLights[%i].vertex2", i);
		shader->setUniform(buffer, l.vertex2);
	}

	for (int i = 0, s = static_cast<int>(m_PointLights.size()); i < s; i++)
	{
		const auto &l = m_PointLights.at(i);

		char buffer[128];
		sprintf(buffer, "pointLights[%i].position_energy", i);
		shader->setUniform(buffer, vec4(l.position, l.energy));
		sprintf(buffer, "pointLights[%i].radiance", i);
		shader->setUniform(buffer, l.radiance);
	}

	for (int i = 0, s = static_cast<int>(m_SpotLights.size()); i < s; i++)
	{
		const auto &l = m_SpotLights.at(i);

		char buffer[128];
		sprintf(buffer, "spotLights[%i].position_cos_inner", i);
		shader->setUniform(buffer, vec4(l.position, l.cosInner));
		sprintf(buffer, "spotLights[%i].radiance_cos_outer", i);
		shader->setUniform(buffer, vec4(l.radiance, l.cosOuter));
		sprintf(buffer, "spotLights[%i].direction_energy", i);
		shader->setUniform(buffer, vec4(l.direction, l.energy));
	}

	for (int i = 0, s = static_cast<int>(m_DirectionalLights.size()); i < s; i++)
	{
		const auto &l = m_DirectionalLights.at(i);

		char buffer[128];
		sprintf(buffer, "dirLights[%i].direction_energy", i);
		shader->setUniform(buffer, vec4(l.direction, l.energy));
		sprintf(buffer, "dirLights[%i].radiance", i);
		shader->setUniform(buffer, l.radiance);
	}

	shader->unbind();
}
