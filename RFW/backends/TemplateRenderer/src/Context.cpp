#include "PCH.h"

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() {}

std::vector<rfw::RenderTarget> Context::get_supported_targets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		auto error = glewInit();
		if (error != GLEW_NO_ERROR)
			throw std::runtime_error("Could not init GLEW.");
		m_InitializedGlew = true;
		CheckGL();
	}

	m_Width = width;
	m_Height = height;
	m_TargetID = *glTextureID;
}

void Context::cleanup() {}

void Context::render_frame(const rfw::Camera &camera, rfw::RenderStatus status) {}

void Context::set_materials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void Context::set_textures(const std::vector<rfw::TextureData> &textures) {}

void Context::set_mesh(size_t index, const rfw::Mesh &mesh) {}

void Context::set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) {}

void Context::set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

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
}

void Context::get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::get_settings() const { return {}; }

void Context::set_setting(const rfw::RenderSetting &setting) {}

void Context::update() {}

void Context::set_probe_index(glm::uvec2 probePos) {}

rfw::RenderStats Context::get_stats() const { return rfw::RenderStats(); }
