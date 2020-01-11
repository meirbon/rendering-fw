#include "PCH.h"
#include "Context.h"

using namespace cl;
using namespace rfw;

rfw::RenderContext *createRenderContext() { return new RTContext(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

RTContext::RTContext()
{
	m_Context = std::make_shared<CLContext>();
	m_DebugKernel = new CLKernel(m_Context, "clkernels/debug.cl", "draw", {512 / 16, 512 / 16, 1}, {16, 16, 1});
}

RTContext::~RTContext() { delete m_Target; }

std::vector<rfw::RenderTarget> RTContext::getSupportedTargets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void RTContext::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void RTContext::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		auto error = glewInit();
		if (error != GLEW_NO_ERROR)
			throw std::runtime_error("Could not init GLEW.");
		m_InitializedGlew = true;
		CheckGL();
	}

	delete m_Target;
	m_Target = new cl::CLBuffer<glm::vec4, BufferType::TARGET>(m_Context, *glTextureID, width, height);

	m_DebugKernel->set_buffer(0, m_Target);
	m_DebugKernel->set_global_size({width, height, 1});

	m_Width = width;
	m_Height = height;
	m_TargetID = *glTextureID;
}

void RTContext::cleanup() {}

void RTContext::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	glFinish();
	m_DebugKernel->run();
	m_Context->finish();
}

void RTContext::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void RTContext::setTextures(const std::vector<rfw::TextureData> &textures) {}

void RTContext::setMesh(size_t index, const rfw::Mesh &mesh) {}

void RTContext::setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) {}

void RTContext::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

void RTContext::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
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

void RTContext::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings RTContext::getAvailableSettings() const { return {}; }

void RTContext::setSetting(const rfw::RenderSetting &setting) {}

void RTContext::update() {}

void RTContext::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats RTContext::getStats() const { return rfw::RenderStats(); }
