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

	m_Accumulator.resize(width * height);

	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	CheckGL();
	std::vector<glm::vec4> dummyData(m_Width * m_Height, glm::vec4(0));
	glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, m_Width * m_Height * sizeof(glm::vec4), dummyData.data(),
				 GL_STREAM_DRAW_ARB);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CheckGL();
}

void Context::cleanup() {}

void Context::render_frame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	if (status == Reset)
	{
		m_Samples = 0;
		memset(m_Accumulator.data(), 0, m_Accumulator.size() * sizeof(vec4));
	}

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
		handles.push_back(m_Pool.push([packetsPerThread, maxPixelID, &probe_id, &maxPixelID4, &maxPixelID8,
									   this](int tID) {
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

				for (int p_id = 0; p_id < PACKET_WIDTH; p_id++)
				{
					const int pixelID = packet.ray.id[p_id];
					if (pixelID >= maxPixelID)
						continue;

					if (pixelID == probe_id)
					{
						m_ProbedDist = packet.ray.tfar[p_id];
						m_ProbedInstance = packet.hit.instID[0][p_id];
						m_ProbedTriangle = packet.hit.primID[p_id];
					}

					int primID = packet.hit.primID[p_id];
					int instID = packet.hit.instID[0][p_id];
					float t = packet.ray.tfar[p_id];
					vec3 origin = vec3(packet.ray.org_x[p_id], packet.ray.org_y[p_id], packet.ray.org_z[p_id]);
					vec3 direction = vec3(packet.ray.dir_x[p_id], packet.ray.dir_y[p_id], packet.ray.dir_z[p_id]);
					vec3 throughput = vec3(1.0f);
					for (int pl = 0; pl < MAX_PATH_LENGTH; pl++)
					{
						if (instID == RTC_INVALID_GEOMETRY_ID || primID == RTC_INVALID_GEOMETRY_ID)
						{
							const vec2 uv =
								vec2(0.5f * (1.0f + glm::atan(direction.x, -direction.z) * glm::one_over_pi<float>()),
									 glm::acos(direction.y) * glm::one_over_pi<float>());
							const uvec2 pUv = uvec2(uv.x * float(m_SkyboxWidth - 1), uv.y *float(m_SkyboxHeight - 1));
							m_Accumulator[pixelID] += vec4(throughput * m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
							break;
						}

						const vec3 p = origin + direction * t;

						const Triangle &tri = m_Meshes[m_InstanceMesh[instID]].triangles[primID];
						const auto &material = m_Materials[tri.material];

						ShadingData shadingData;
						shadingData.color = material.getColor();

						if (any(greaterThan(shadingData.color, vec3(1.0f))))
						{
							m_Accumulator[pixelID] += vec4(throughput * shadingData.color, 0.0f);
							break;
						}

						const simd::matrix4 &matrix = m_InstanceMatrices[instID];
						const mat3 &normal_matrix = m_InverseMatrices[instID];
						vec3 N = normalize(normal_matrix * vec3(tri.Nx, tri.Ny, tri.Nz));

						const simd::vector4 vertex0_4 =
							matrix * simd::vector4(tri.vertex0.x, tri.vertex0.y, tri.vertex0.z, 1.0f);
						const simd::vector4 vertex1_4 =
							matrix * simd::vector4(tri.vertex1.x, tri.vertex1.y, tri.vertex1.z, 1.0f);
						const simd::vector4 vertex2_4 =
							matrix * simd::vector4(tri.vertex2.x, tri.vertex2.y, tri.vertex2.z, 1.0f);

						const vec3 vertex0 = vec3(vertex0_4.vec);
						const vec3 vertex1 = vec3(vertex1_4.vec);
						const vec3 vertex2 = vec3(vertex2_4.vec);

						const vec3 bary = triangle::getBaryCoords(p, N, vertex0, vertex1, vertex2);

						vec3 iN =
							normalize(normal_matrix * vec3(bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2));


						m_Accumulator[pixelID] += vec4(iN, 0.0f);
						const float flip = -sign(dot(direction, N));
						N *= flip;	// Fix geometric normal
						iN *= flip; // Fix interpolated normal

						vec3 T, B;
						createTangentSpace(iN, T, B);

						uint seed = WangHash(pixelID * 16789 + m_Samples * 1791 + pl * 720898027);

						shadingData.parameters = material.parameters;
						shadingData.matID = tri.material;
						shadingData.absorption = material.getAbsorption();

						float tu, tv;
						if (material.hasFlag(HasDiffuseMap) || material.hasFlag(HasNormalMap) ||
							material.hasFlag(HasRoughnessMap) || material.hasFlag(HasAlphaMap) ||
							material.hasFlag(HasSpecularityMap))
						{
							tu = bary.x * tri.u0 + bary.y * tri.u1 + bary.z * tri.u2;
							tv = bary.x * tri.v0 + bary.y * tri.v1 + bary.z * tri.v2;
						}

						if (material.hasFlag(HasDiffuseMap))
						{
							const float u = (tu + material.uoffs0) * material.uscale0;
							const float v = (tv + material.voffs0) * material.vscale0;

							float tx = fmod(u, 1.0f);
							float ty = fmod(v, 1.0f);

							if (tx < 0.f)
								tx = 1.f + tx;
							if (ty < 0.f)
								ty = 1.f + ty;

							const auto &tex = m_Textures[material.texaddr0];

							const uint ix = uint(tx * (tex.width - 1));
							const uint iy = uint(ty * (tex.height - 1));
							const auto texelID = static_cast<int>(iy * tex.width + ix);

							switch (tex.type)
							{
							case (TextureData::FLOAT4):
							{
								shadingData.color =
									shadingData.color * vec3(reinterpret_cast<vec4 *>(tex.data)[texelID]);
							}
							case (TextureData::UINT):
							{
								// RGBA
								const uint texel = reinterpret_cast<uint *>(tex.data)[texelID];
								constexpr float tscale = 1.0f / 256.0f;
								shadingData.color = shadingData.color * tscale *
													vec3(texel & 0xFFu, (texel >> 8u) & 0xFFu, (texel >> 16u) & 0xFFu);
							}
							}
						}

						bool specular = false;
						vec3 R;
						float pdf;
						const vec3 bsdf =
							SampleBSDF(shadingData, iN, N, T, B, -direction, t, bool(flip < 0), direction, pdf, seed);

						throughput = throughput * bsdf * glm::abs(glm::dot(iN, R)) * (1.0f / pdf);
						if (any(isnan(throughput)))
							break;

						origin = p + 1e-6f * R;
						direction = R;

						RTCRayHit hit = {};
						hit.ray.tnear = 1e-5f;
						hit.ray.tfar = 1e26f;
						hit.ray.org_x = origin.x;
						hit.ray.org_y = origin.y;
						hit.ray.org_z = origin.z;

						hit.ray.dir_x = direction.x;
						hit.ray.dir_y = direction.y;
						hit.ray.dir_z = direction.z;

						rtcIntersect1(m_Scene, &context, &hit);
						t = hit.ray.tfar;
						instID = hit.hit.instID[0];
						primID = hit.hit.geomID;
					}
				}
			}
		}));
	}

	for (auto &handle : handles)
		handle.get();
	m_Samples++;
	handles.clear();

	const float scale = 1.0f / float(m_Samples);
	for (int i = 0, s = threads; i < s; i++)
	{
		handles.push_back(m_Pool.push([this, &threads, &scale](int tId) {
			for (int y = tId; y < m_Height; y += threads)
			{
				const uint y_offset = y * m_Width;
				for (int x = 0; x < m_Width; x++)
				{
					const int pixel_id = y_offset + x;
					m_Pixels[pixel_id] = m_Accumulator[pixel_id] * scale;
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

void Context::set_materials(const std::vector<rfw::DeviceMaterial> &materials,
							const std::vector<rfw::MaterialTexIds> &texDescriptors)
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
		rtcSetGeometryTransform(rtcGetGeometry(m_Scene, m_Instances[i]), 0, RTC_FORMAT_FLOAT4X4_COLUMN_MAJOR,
								value_ptr(transform));
		rtcCommitGeometry(instance);
	}

	m_InstanceMesh[i] = meshIdx;
	m_InstanceMatrices[i] = transform;
	m_InverseMatrices[i] = inverse_transform;
}

void Context::set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	m_Skybox = pixels;
	m_SkyboxWidth = int(width);
	m_SkyboxHeight = int(height);
}

void Context::set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
						 const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
						 const rfw::DeviceDirectionalLight *directionalLights)
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
