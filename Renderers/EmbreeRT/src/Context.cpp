#include "Context.h"

#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>
#include <utils/Timer.h>
#include <utils/gl/CheckGL.h>

#include <utils/Concurrency.h>
#include <utils/Xor128.h>

#ifdef _WIN32
#include <ppl.h>
#endif

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

#define PACKET_WIDTH 8

Context::~Context()
{
	glDeleteBuffers(1, &m_PboID);
	rtcReleaseScene(m_Scene);
	m_Scene = nullptr;
	rtcReleaseDevice(m_Device);
	m_Device = nullptr;
}

std::vector<rfw::RenderTarget> Context::getSupportedTargets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void rtcErrorFunc(void *userPtr, enum RTCError code, const char *str)
{
	WARNING("Error callback: %s", str);
	switch (code)
	{
	case RTC_ERROR_NONE:
		break;
	case RTC_ERROR_UNKNOWN:
		WARNING("Embree: unkown error");
	case RTC_ERROR_INVALID_ARGUMENT:
		WARNING("Embree: invalid argument");
	case RTC_ERROR_INVALID_OPERATION:
		WARNING("Embree: invalid operation");
	case RTC_ERROR_OUT_OF_MEMORY:
		WARNING("Embree: out of memory");
	case RTC_ERROR_UNSUPPORTED_CPU:
		WARNING("Embree: unsupported CPU");
	case RTC_ERROR_CANCELLED:
		WARNING("Embree: error cancelled");
	}
}

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		std::vector<char> config(512, 0);
		utils::string::format(config.data(), "threads=%ul", std::thread::hardware_concurrency());
		m_Device = rtcNewDevice(config.data());
		m_Scene = rtcNewScene(m_Device);
		rtcSetDeviceErrorFunction(m_Device, rtcErrorFunc, nullptr);

		const auto error = glewInit();
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
	CheckGL();

	const auto camParams = Ray::CameraParams(camera.getView(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	auto timer = utils::Timer();

#if PACKET_WIDTH == 4
	constexpr int TILE_WIDTH = 2;
	constexpr int TILE_HEIGHT = 2;
	const int wTiles = m_Width / TILE_WIDTH + 1;
	const int hTiles = m_Height / TILE_HEIGHT + 1;

	std::vector<RTCRayHit4> packets(wTiles * hTiles);
#elif PACKET_WIDTH == 8
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 2;
	const int wTiles = m_Width / TILE_WIDTH + 1;
	const int hTiles = m_Height / TILE_HEIGHT + 1;

	std::vector<RTCRayHit8> packets(wTiles * hTiles);
#elif PACKET_WIDTH == 16
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 4;
	const int wTiles = m_Width / TILE_WIDTH + 1;
	const int hTiles = m_Height / TILE_HEIGHT + 1;

	std::vector<RTCRayHit16> packets(wTiles * hTiles);
#endif

	const int maxPixelID = m_Width * m_Height;

	for (int tile_y = 0; tile_y < hTiles; tile_y++)
	{
		for (int tile_x = 0; tile_x < wTiles; tile_x++)
		{
#if PACKET_WIDTH == 4
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x4[4] = {x, x + 1, x, x + 1};
			const int y4[4] = {y, y, y + 1, y + 1};
			packets[tile_id] = Ray::GenerateRay4(camParams, x4, y4, &m_Rng);
#elif PACKET_WIDTH == 8

			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x8[8] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
			const int y8[8] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1};
			packets[tile_id] = Ray::GenerateRay8(camParams, x8, y8, &m_Rng);
#elif PACKET_WIDTH == 16
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x16[16] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
			const int y16[16] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1, y + 2, y + 2, y + 2, y + 2, y + 3, y + 3, y + 3, y + 3};

			const int y16[16] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1, y + 2, y + 2, y + 2, y + 2, y + 3, y + 3, y + 3, y + 3};
			packets[tile_id] = Ray::GenerateRay16(camParams, x16, y16, &m_Rng);
#endif
		}
	}

	const auto threads = m_Pool.size();
	const auto packetsPerThread = packets.size() / threads;
	std::vector<std::future<void>> handles(threads);

	for (size_t i = 0; i < threads; i++)
	{
		handles[i] = m_Pool.push([i, packetsPerThread, &packets, this](int tID) {
			const int start = static_cast<int>(i * packetsPerThread);
			const int end = static_cast<int>((i + 1) * packetsPerThread);
			const int valid[8] = {-1, -1, -1, -1, -1, -1, -1, -1};

			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

			for (int i = start; i < end; i++)
			{
#if PACKET_WIDTH == 4
				rtcIntersect4(valid, m_Scene, &context, &packets[i]);
#elif PACKET_WIDTH == 8
				rtcIntersect8(valid, m_Scene, &context, &packets[i]);
#elif PACKET_WIDTH == 16
				rtcIntersect16(valid, m_Scene, &context, &packets[i]);
#endif
			}
		});
	}

	for (int i = 0, s = static_cast<int>(threads); i < s; i++)
		handles[i].get();

	for (size_t i = 0; i < threads; i++)
	{
		handles[i] = m_Pool.push([i, packetsPerThread, maxPixelID, &packets, this](int tID) {
			const int start = static_cast<int>(i * packetsPerThread);
			const int end = static_cast<int>((i + 1) * packetsPerThread);

			for (int i = start; i < end; i++)
			{
				const auto &packet = packets[i];
				for (int i = 0; i < PACKET_WIDTH; i++)
				{
					const int pixel_id = packet.ray.id[i];
					if (pixel_id >= maxPixelID)
						break;

					if (packet.hit.geomID[i] == RTC_INVALID_GEOMETRY_ID)
					{
						const vec2 uv = vec2(0.5f * (1.0f + atan(packet.ray.dir_x[i], -packet.ray.dir_z[i]) * glm::one_over_pi<float>()),
											 acos(packet.ray.dir_y[i]) * glm::one_over_pi<float>());
						const ivec2 pUv = ivec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
						const int skyboxPixel = pUv.y * m_SkyboxWidth + pUv.x;
						m_Pixels[pixel_id] = vec4(m_Skybox[skyboxPixel], 0.0f);
						continue;
					}

					const auto u = packet.hit.u[i];
					const auto v = packet.hit.v[i];
					const auto w = 1.0f - u - v;

					m_Pixels[pixel_id] = vec4(u, v, w, 0.0f);
				}
			}
		});
	}

	for (int i = 0, s = static_cast<int>(threads); i < s; i++)
		handles[i].get();

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

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
	m_Materials.resize(materials.size());
	memcpy(m_Materials.data(), materials.data(), materials.size() * sizeof(Material));
}

void Context::setTextures(const std::vector<rfw::TextureData> &textures) { m_Textures = textures; }

void Context::setMesh(size_t index, const rfw::Mesh &mesh)
{
	if (index >= m_Meshes.size())
	{
		while (index >= m_Meshes.size())
		{
			m_MeshChanged.emplace_back(false);
			m_Meshes.emplace_back(m_Device);
		}
	}

	m_MeshChanged[index] = true;
	m_Meshes[index].setGeometry(mesh);
}

void Context::setInstance(const size_t i, const size_t meshIdx, const mat4 &transform)
{
	if (m_Instances.size() <= i)
	{
		m_Instances.emplace_back(0);
		m_InstanceMesh.emplace_back(0);
		m_InstanceMatrices.emplace_back();
		m_InverseMatrices.emplace_back();

		const auto instance = rtcNewGeometry(m_Device, RTC_GEOMETRY_TYPE_INSTANCE);
		rtcSetGeometryInstancedScene(instance, m_Meshes[meshIdx].scene);
		rtcSetGeometryTimeStepCount(instance, 1);
		rtcSetGeometryTransform(instance, 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, value_ptr(transform));
		rtcCommitGeometry(instance);
		m_Instances[i] = rtcAttachGeometry(m_Scene, instance);
	}
	else
	{
		auto instance = rtcGetGeometry(m_Scene, m_Instances[i]);
		rtcSetGeometryInstancedScene(instance, m_Meshes[meshIdx].scene);
		rtcSetGeometryTimeStepCount(instance, 1);
		rtcSetGeometryTransform(rtcGetGeometry(m_Scene, m_Instances[i]), 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR, value_ptr(transform));
		rtcCommitGeometry(instance);
	}

	m_InstanceMesh[i] = meshIdx;
	m_InstanceMatrices[i] = transform;
	m_InverseMatrices[i] = transpose(inverse(mat3(transform)));
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

void Context::update()
{
	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		if (!m_MeshChanged[i])
			continue;

		auto instance = rtcGetGeometry(m_Scene, m_Instances[i]);
		rtcSetGeometryInstancedScene(instance, m_Meshes[m_InstanceMesh[i]].scene);
		rtcCommitGeometry(instance);
	}

	rtcCommitScene(m_Scene);
}

void Context::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats Context::getStats() const { return m_Stats; }
