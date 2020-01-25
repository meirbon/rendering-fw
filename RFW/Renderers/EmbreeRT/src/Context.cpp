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

Context::~Context()
{
	glDeleteBuffers(1, &m_PboID);
	rtcReleaseScene(m_Scene);
	m_Scene = nullptr;
	rtcReleaseDevice(m_Device);
	m_Device = nullptr;
}

std::vector<rfw::RenderTarget> Context::get_supported_targets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

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

void Context::render_frame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	glFinish();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);
	CheckGL();

	const auto camParams = Ray::CameraParams(camera.get_view(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	auto timer = utils::Timer();

	const int wTiles = (m_Width + m_Width % TILE_WIDTH) / TILE_WIDTH;
	const int hTiles = (m_Height + m_Height % TILE_HEIGHT) / TILE_HEIGHT;

	m_Packets.resize(wTiles * hTiles);

	utils::concurrency::parallel_for(0, hTiles, [&](int tile_y) {
		for (int tile_x = 0; tile_x < wTiles; tile_x++)
		{
#if PACKET_WIDTH == 4
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x4[4] = {x, x + 1, x, x + 1};
			const int y4[4] = {y, y, y + 1, y + 1};
			m_Packets[tile_id] = Ray::GenerateRay4(camParams, x4, y4, &m_Rng);
#elif PACKET_WIDTH == 8

			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x8[8] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
			const int y8[8] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1};
			m_Packets[tile_id] = Ray::GenerateRay8(camParams, x8, y8, &m_Rng);
#elif PACKET_WIDTH == 16
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x16[16] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
			const int y16[16] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1, y + 2, y + 2, y + 2, y + 2, y + 3, y + 3, y + 3, y + 3};

			const int y16[16] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1, y + 2, y + 2, y + 2, y + 2, y + 3, y + 3, y + 3, y + 3};
			m_Packets[tile_id] = Ray::GenerateRay16(camParams, x16, y16, &m_Rng);
#endif
		}
	});

	const auto threads = m_Pool.size();
	const auto packetsPerThread = m_Packets.size() / threads;

	const int maxPixelID = m_Width * m_Height;
	const __m128i maxPixelID4 = _mm_set1_epi32(maxPixelID - 1);
	const __m256i maxPixelID8 = _mm256_set1_epi32(maxPixelID - 1);
	const int probe_id = m_ProbePos.y * m_Width + m_ProbePos.x;

	std::vector<std::future<void>> handles;
	handles.reserve(threads);

	for (int i = 0, s = static_cast<int>(threads); i < s; i++)
	{
		handles.push_back(m_Pool.push([packetsPerThread, maxPixelID, &probe_id, &maxPixelID4, &maxPixelID8, this](int tID) {
			const int start = static_cast<int>(tID * packetsPerThread);
			const int end = static_cast<int>((tID + 1) * packetsPerThread);

			RTCIntersectContext context;
			rtcInitIntersectContext(&context);
			context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
			int valid[PACKET_WIDTH];
			memset(valid, -1, sizeof(valid));

			for (int i = start; i < end; i++)
			{
				const auto &packet = m_Packets[i];
#if PACKET_WIDTH == 4
				rtcIntersect4(valid, m_Scene, &context, &m_Packets[i]);
#elif PACKET_WIDTH == 8
				rtcIntersect8(valid, m_Scene, &context, &m_Packets[i]);
#elif PACKET_WIDTH == 16
				rtcIntersect16(valid, m_Scene, &context, &m_Packets[i]);
#endif

				const simd::vector4 one_over_pi = _mm_set1_ps(glm::one_over_pi<float>());
				const simd::vector4 one4 = _mm_set1_ps(1.0f);
				const simd::vector4 min_one4 = _mm_set1_ps(-1.0f);
				const simd::vector4 half4 = _mm_set1_ps(0.5f);
				const simd::vector4 zero4 = _mm_setzero_ps();
				const simd::vector4 skybox_width = _mm_set1_ps(float(m_SkyboxWidth - 1));
				const simd::vector4 skybox_height = _mm_set1_ps(float(m_SkyboxHeight - 1));
				const __m128i invalid_geometry_id = _mm_set1_epi32(RTC_INVALID_GEOMETRY_ID);

				for (int k = 0; k < PACKET_WIDTH; k += 4)
				{
					const __m128i mask = _mm_cmpgt_epi32(reinterpret_cast<const __m128i &>(*(packet.ray.id + k)), maxPixelID4);
					const int pixel_mask = _mm_movemask_ps(_mm_castsi128_ps(mask));
					if (pixel_mask == 15)
						continue;

					const __m128i invalid_mask = _mm_cmpeq_epi32(reinterpret_cast<const __m128i &>(*(packet.hit.geomID + k)), invalid_geometry_id);
					if (_mm_movemask_ps(_mm_castsi128_ps(invalid_mask)) > 0)
					{
						const auto &dir_x = reinterpret_cast<const simd::vector4 &>(*(packet.ray.dir_x + k));
						const auto &dir_y = reinterpret_cast<const simd::vector4 &>(*(packet.ray.dir_y + k));
						const auto &dir_z = reinterpret_cast<const simd::vector4 &>(*(packet.ray.dir_z + k));

						const simd::vector4 u4 = half4 * (one4 + rfw::simd::atan2(dir_x, dir_z * min_one4) * one_over_pi);
						const simd::vector4 v4 = rfw::simd::acos(dir_y) * one_over_pi;

						const auto p_u4 = _mm_round_ps((min(one4, max(zero4, u4)) * skybox_width).vec_4, _MM_FROUND_TO_NEAREST_INT);
						const simd::vector4 p_v4 =
							simd::vector4(_mm_round_ps((min(one4, max(zero4, v4)) * skybox_height).vec_4, _MM_FROUND_TO_NEAREST_INT)) * skybox_width;

						const auto *pu = reinterpret_cast<const float *>(&p_u4);
						const auto *pv = reinterpret_cast<const float *>(&p_v4);

						for (int m = 0; m < 4; m++)
						{
							if (((pixel_mask >> m) & 1u) == 1)
								continue;
							m_Pixels[packet.ray.id[k + m]] = vec4(m_Skybox[int(pv[m]) + int(pu[m])], 0.0f);
						}
					}
				}

				for (int j = 0; j < PACKET_WIDTH; j++)
				{
					const int pixel_id = packet.ray.id[j];
					if (pixel_id >= maxPixelID)
						continue;
					if (packet.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID)
						continue;

					const int instID = packet.hit.instID[0][j];
					const int primID = packet.hit.primID[j];

					if (pixel_id == probe_id)
					{
						m_ProbedDist = packet.ray.tfar[j];
						m_ProbedInstance = instID;
						m_ProbedTriangle = primID;
					}

					const mat3 &invTransform = m_InverseMatrices[instID];
					const Triangle *tri = &m_Meshes[m_InstanceMesh[instID]].triangles[primID];

					const vec3 bary = vec3(1.0f - packet.hit.u[j] - packet.hit.v[j], packet.hit.u[j], packet.hit.v[j]);

					const vec3 N = invTransform * vec3(tri->Nx, tri->Ny, tri->Nz);
					const vec3 iN = normalize(invTransform * (bary.z * tri->vN0 + bary.x * tri->vN1 + bary.y * tri->vN2));

					const Material &material = m_Materials[tri->material];
					vec3 color = material.getColor();

					float tu, tv;
					if (material.hasFlag(HasDiffuseMap) || material.hasFlag(HasNormalMap) || material.hasFlag(HasRoughnessMap) ||
						material.hasFlag(HasAlphaMap) || material.hasFlag(HasSpecularityMap))
					{
						tu = bary.x * tri->u0 + bary.y * tri->u1 + bary.z * tri->u2;
						tv = bary.x * tri->v0 + bary.y * tri->v1 + bary.z * tri->v2;
					}

					if (material.hasFlag(HasDiffuseMap))
					{
						const float u = (tu + material.uoffs0) * material.uscale0;
						const float v = (tv + material.voffs0) * material.vscale0;

						float x = fmod(u, 1.0f);
						float y = fmod(v, 1.0f);

						if (x < 0)
							x = 1 + x;
						if (y < 0)
							y = 1 + y;

						const auto &tex = m_Textures[material.texaddr0];

						const uint ix = uint(x * (tex.width - 1));
						const uint iy = uint(y * (tex.height - 1));
						const auto texelID = static_cast<int>(iy * tex.width + ix);

						switch (tex.type)
						{
						case (TextureData::FLOAT4):
						{
							color = color * vec3(reinterpret_cast<vec4 *>(tex.data)[texelID]);
						}
						case (TextureData::UINT):
						{
							// RGBA
							const uint texel = reinterpret_cast<uint *>(tex.data)[texelID];
							constexpr float s = 1.0f / 256.0f;

							color = color * s * vec3(texel & 0xFFu, (texel >> 8u) & 0xFFu, (texel >> 16u) & 0xFFu);
						}
						}
					}

					m_Pixels[pixel_id] = vec4(color, 0.0f);
				}
			}
		}));
	}

	for (auto &handle : handles)
		handle.get();

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

	glFinish();
}

void Context::set_materials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
	m_Materials.resize(materials.size());
	memcpy(m_Materials.data(), materials.data(), materials.size() * sizeof(Material));
}

void Context::set_textures(const std::vector<rfw::TextureData> &textures) { m_Textures = textures; }

void Context::set_mesh(size_t index, const rfw::Mesh &mesh)
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

void Context::set_instance(const size_t i, const size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
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
	m_InverseMatrices[i] = inverse_transform;
}

void Context::set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	m_Skybox = pixels;
	m_SkyboxWidth = width;
	m_SkyboxHeight = height;
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
}

void Context::get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const
{
	*instanceIndex = m_ProbedInstance;
	*primitiveIndex = m_ProbedTriangle;
	*distance = m_ProbedDist;
}

rfw::AvailableRenderSettings Context::get_settings() const { return {}; }

void Context::set_setting(const rfw::RenderSetting &setting) {}

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

void Context::set_probe_index(glm::uvec2 probePos) { m_ProbePos = probePos; }

rfw::RenderStats Context::get_stats() const { return m_Stats; }
