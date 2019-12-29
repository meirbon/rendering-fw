#include "Context.h"

#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>
#include <utils/Timer.h>
#include <utils/gl/CheckGL.h>

#include <utils/Xor128.h>

#ifdef _WIN32
#include <ppl.h>
#endif

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

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

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		std::vector<char> config(512, 0);
		utils::string::format(config.data(), "threads=%ul", std::thread::hardware_concurrency());
		m_Device = rtcNewDevice(config.data());
		m_Scene = rtcNewScene(m_Device);

		// TODO: Create context for coherent and incoherent rays
		rtcInitIntersectContext(&m_PrimaryContext);
		m_PrimaryContext.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;

		rtcInitIntersectContext(&m_SecondaryContext);
		m_SecondaryContext.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;

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
	// utils::Xor128 rng = {};
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);
	CheckGL();

	const auto camParams = Ray::CameraParams(camera.getView(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	auto timer = utils::Timer();

	constexpr int TILE_WIDTH = 2;
	constexpr int TILE_HEIGHT = 2;
	const int maxPixelID = m_Width * m_Height;

	const float inv_width = 1.0f / float(camParams.scrwidth);
	const float inv_height = 1.0f / float(camParams.scrheight);

	const vec3 p1 = camParams.p1;
	const vec3 right = camParams.right_spreadAngle;
	const vec3 up = camParams.up;
	const vec3 baseOrigin = vec3(camParams.pos_lensSize);

	for (int tile_y = 0, hTiles = m_Height / TILE_HEIGHT; tile_y < hTiles; tile_y++)
	{
		for (int tile_x = 0, wTiles = m_Width / TILE_WIDTH; tile_x < wTiles; tile_x++)
		{
#if 1
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;

			const int x4[4] = {x, x + 1, x, x + 1};
			const int y4[4] = {y, y, y + 1, y + 1};
			RTCRayHit4 query = Ray::GenerateRay4(camParams, x4, y4, &m_Rng);

			int valid[4];
			memset(valid, -1, sizeof(valid));
			rtcIntersect4(valid, m_Scene, &m_PrimaryContext, &query);

			for (int i = 0; i < 4; i++)
			{
				const int pixel_id = query.ray.id[i];
				if (pixel_id > maxPixelID)
					break;

				if (query.hit.geomID[i] == RTC_INVALID_GEOMETRY_ID)
				{
					const vec2 uv = vec2(0.5f * (1.0f + atan(query.ray.dir_x[i], -query.ray.dir_z[i]) * glm::one_over_pi<float>()),
										 acos(query.ray.dir_y[i]) * glm::one_over_pi<float>());
					const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
					m_Pixels[pixel_id] = vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
					continue;
				}

				const auto u = query.hit.u[i];
				const auto v = query.hit.v[i];
				const auto w = 1.0f - u - v;

				m_Pixels[pixel_id] = vec4(u, v, w, 0.0f);
			}
#else
			RTCRayHit4 query4;
			for (int i = 0; i < 4; i++)
			{
				query4.ray.tfar[i] = 1e34f;
				query4.ray.tnear[i] = 1e-5f;
				query4.hit.geomID[i] = RTC_INVALID_GEOMETRY_ID;
				query4.hit.primID[i] = RTC_INVALID_GEOMETRY_ID;
				query4.hit.instID[0][i] = RTC_INVALID_GEOMETRY_ID;
			}

			for (int y = 0; y < TILE_HEIGHT; y++)
			{
				const int local_y = tile_y * TILE_HEIGHT;
				const int pixel_y = y + local_y;
				const int y_offset = pixel_y * m_Height;

				for (int x = 0; x < TILE_WIDTH; x++)
				{
					const int local_x = tile_x * TILE_WIDTH;
					const int pixel_x = x + local_x;

					const int local_id = x + y * TILE_WIDTH;

					float r0 = m_Rng.Rand();
					float r1 = m_Rng.Rand();
					float r2 = m_Rng.Rand();
					float r3 = m_Rng.Rand();

					const float blade = float(int(r0 * 9));
					r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
					float x1, y1, x2, y2;
					constexpr float piOver4point5 = 3.14159265359f / 4.5f;
					float bladeParam = blade * piOver4point5;
					x1 = cos(bladeParam);
					y1 = sin(bladeParam);
					bladeParam = (blade + 1.0f) * piOver4point5;
					x2 = cos(bladeParam);
					y2 = sin(bladeParam);
					if ((r2 + r3) > 1.0f)
					{
						r2 = 1.0f - r2;
						r3 = 1.0f - r3;
					}

					const float xr = x1 * r2 + x2 * r3;
					const float yr = y1 * r2 + y2 * r3;

					const vec3 origin = baseOrigin + camParams.pos_lensSize.w * (right * xr + up * yr);
					const float u = (float(pixel_x) + r0) * inv_width;
					const float v = (float(pixel_y) + r1) * inv_height;
					const vec3 pointOnPixel = p1 + u * right + v * up;
					const vec3 direction = normalize(pointOnPixel - origin);

					query4.ray.org_x[local_id] = origin.x;
					query4.ray.org_y[local_id] = origin.y;
					query4.ray.org_z[local_id] = origin.z;
					query4.ray.id[local_id] = pixel_x + pixel_y * camParams.scrwidth;
					query4.ray.dir_x[local_id] = direction.x;
					query4.ray.dir_y[local_id] = direction.y;
					query4.ray.dir_z[local_id] = direction.z;
				}
			}

			int valid[16];
			memset(valid, -1, sizeof(valid));
			rtcIntersect4(valid, m_Scene, &m_PrimaryContext, &query4);

			for (int i = 0; i < 4; i++)
			{
				const int pixel_id = query4.ray.id[i];
				if (pixel_id > maxPixelID)
					break;

				if (query4.hit.geomID[i] == RTC_INVALID_GEOMETRY_ID)
				{
					const vec2 uv = vec2(0.5f * (1.0f + atan(query4.ray.dir_x[i], -query4.ray.dir_z[i]) * glm::one_over_pi<float>()),
										 acos(query4.ray.dir_y[i]) * glm::one_over_pi<float>());
					const ivec2 pUv = ivec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
					const int skyboxPixel = pUv.y * m_SkyboxWidth + pUv.x;
					m_Pixels[pixel_id] = vec4(m_Skybox[skyboxPixel], 0.0f);
					continue;
				}

				const auto u = query4.hit.u[i];
				const auto v = query4.hit.v[i];
				const auto w = 1.0f - u - v;

				m_Pixels[pixel_id] = vec4(u, v, w, 0.0f);
			}

#endif
		}
	}

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
