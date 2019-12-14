#include "Context.h"

#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>
#include <utils/Timer.h>

#ifdef _WIN32
#include <ppl.h>
#endif

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() { glDeleteBuffers(1, &m_PboID); }

std::vector<rfw::RenderTarget> Context::getSupportedTargets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

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
		glGenBuffers(1, &m_PboID);
	}

	m_Width = static_cast<int>(width);
	m_Height = static_cast<int>(height);
	m_TargetID = *glTextureID;

	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	CheckGL();
	std::vector<glm::vec4> dummyData(m_Width * m_Height, glm::vec4(0));
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, m_Width * m_Height * sizeof(glm::vec4), dummyData.data(), GL_STREAM_DRAW_ARB);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CheckGL();
}

void Context::cleanup() {}

void Context::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);

	const auto camParams = Ray::CameraParams(camera.getView(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	auto timer = rfw::utils::Timer();
#ifdef _WIN32
	concurrency::parallel_for(0, m_Height, [&](const int y) {
#else
	for (int y = 0; y < m_Height; y++)
	{
#endif
		const int yOffset = y * m_Width;

		for (int x = 0; x < m_Width; x++)
		{
			const int pixelIdx = yOffset + x;

			Ray ray = Ray::generateFromView(camParams, x, y, 0, 0, 0, 0);
			const auto result = topLevelBVH.intersect(ray, 1e-5f);

			if (result.has_value())
			{
				const auto &tri = result.value();
				const vec3 N = vec3(tri.Nx, tri.Ny, tri.Nz);
				const vec3 p = ray.origin + ray.direction * ray.t;
				const vec3 bary = triangle::getBaryCoords(p, N, tri.vertex0, tri.vertex1, tri.vertex2);
				const vec3 iN = normalize(bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2);
				m_Pixels[pixelIdx] = glm::vec4(iN, 1.0f);
			}
			else
			{
				const vec2 uv = vec2(0.5f * (1.0f + atan(ray.direction.x, -ray.direction.z) * glm::one_over_pi<float>()),
									 acos(ray.direction.y) * glm::one_over_pi<float>());
				const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
				m_Pixels[pixelIdx] = glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
			}
		}
#ifdef _WIN32
	});
#else
	}
#endif
	m_Stats.primaryTime = timer.elapsed();

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	CheckGL();
	glBindTexture(GL_TEXTURE_2D, m_TargetID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_FLOAT, 0);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CheckGL();
}

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void Context::setTextures(const std::vector<rfw::TextureData> &textures) {}

void Context::setMesh(size_t index, const rfw::Mesh &mesh)
{
	if (index >= m_Meshes.size())
		m_Meshes.emplace_back();

	m_Meshes[index].setGeometry(mesh);
}

void Context::setInstance(size_t i, size_t meshIdx, const mat4 &transform)
{
	topLevelBVH.setInstance(i, transform, &m_Meshes[meshIdx], m_Meshes[meshIdx].mbvh->aabb);
}

void Context::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	m_Skybox = pixels;
	m_SkyboxWidth = width;
	m_SkyboxHeight = height;
}

void Context::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
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

void Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::getAvailableSettings() const { return {}; }

void Context::setSetting(const rfw::RenderSetting &setting) {}

void Context::update() { topLevelBVH.constructBVH(); }

void Context::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats Context::getStats() const { return m_Stats; }
