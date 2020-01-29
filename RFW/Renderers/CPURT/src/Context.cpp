#include "PCH.h"

#define PACKET_TRAVERSAL 1

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() { glDeleteBuffers(1, &m_PboID); }

std::vector<rfw::RenderTarget> Context::get_supported_targets() const { return {OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	m_Pool.clearQueue();

	if (!m_InitializedGlew)
	{
		m_Handles.resize(m_Pool.size());
		m_RNGs.resize(m_Pool.size());
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::Xor128();

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

	if (m_Accumulator.size() < width * height)
	{
		// Prevent reallocating memory often
		m_Accumulator.resize(width * height * 4);
	}

	memset(m_Accumulator.data(), 0, m_Accumulator.size() * sizeof(vec4));
	m_Samples = 0;

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
	using namespace simd;

	if (status == Reset)
	{
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::Xor128();

		m_Samples = 0;
		memset(m_Accumulator.data(), 0, m_Accumulator.size() * sizeof(vec4));
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);

	const auto camParams = cpurt::Ray::CameraParams(camera.get_view(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	const auto timer = utils::Timer();

	const auto s = static_cast<int>(m_Pool.size());

#if PACKET_WIDTH == 4
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 1;
#elif PACKET_WIDTH == 8
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 2;
#endif

	const int wTiles = m_Width / TILE_WIDTH;
	const int hTiles = m_Height / TILE_HEIGHT;
	DEBUG("wTiles: %i, hTiles: %i", wTiles, hTiles);
	const int threads = int(m_Pool.size());
	const int nr_packets = wTiles * hTiles;
	m_Packets.resize(nr_packets);

	std::vector<std::future<void>> handles(threads);

	for (int i = 0; i < threads; i++)
	{
		handles[i] = m_Pool.push([this, &camParams, &threads, &wTiles, &hTiles, &TILE_WIDTH, &TILE_HEIGHT](int t_id) {
			for (int tile_y = t_id; tile_y < hTiles; tile_y += threads)
			{
				for (int tile_x = 0; tile_x < wTiles; tile_x++)
				{
#if PACKET_WIDTH == 4
					const int y = tile_y * TILE_HEIGHT;
					const int x = tile_x * TILE_WIDTH;
					const int tile_id = tile_y * wTiles + tile_x;

#if 1
					const int x4[4] = {x, x + 1, x + 2, x + 3};
					const int y4[4] = {y, y, y, y};
#else
					const int x4[4] = {x, x + 1, x, x + 1};
					const int y4[4] = {y, y, y + 1, y + 1};
#endif
					m_Packets[tile_id] = cpurt::Ray::generateRay4(camParams, x4, y4, &m_RNGs[t_id]);
#elif PACKET_WIDTH == 8
					const int y = tile_y * TILE_HEIGHT;
					const int x = tile_x * TILE_WIDTH;
					const int tile_id = tile_y * wTiles + tile_x;

					const int x8[8] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
					const int y8[8] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1};
					m_Packets[tile_id] = cpurt::Ray::GenerateRay8(camParams, x8, y8, &m_RNGs[t_id]);
#endif
				}
			}
		});
	}

	for (int i = 0; i < threads; i++)
		handles[i].get();

	const int probe_id = m_ProbePos.y * m_Width + m_ProbePos.x;
	const int maxPixelID = m_Width * m_Height;
	const int packetsPerThread = nr_packets / threads;
#if 0
	for (int j = 0; j < threads; j++)
	{
		const int t_id = j;
		handles[j] = m_Pool.push([this, packetsPerThread, nr_packets, maxPixelID, t_id](int) {
			const int start = int(t_id * packetsPerThread);
			const int end = min(int((t_id + 1) * packetsPerThread), nr_packets);

			for (int i = start; i < end; i++)
			{
				auto &packet = m_Packets[i];
				topLevelBVH.intersect4(packet.origin_x, packet.origin_y, packet.origin_z, packet.direction_x,
									   packet.direction_y, packet.direction_z, packet.t, packet.primID, packet.instID,
									   1e-5f);

				for (int p_id = 0, s = TILE_WIDTH * TILE_HEIGHT; p_id < s; p_id++)
				{
					const int pixelID = packet.pixelID[p_id];
					int instID = packet.instID[p_id];
					int primID = packet.primID[p_id];

					const vec3 direction =
						vec3(packet.direction_x[p_id], packet.direction_y[p_id], packet.direction_z[p_id]);
					const vec3 origin = vec3(packet.origin_x[p_id], packet.origin_y[p_id], packet.origin_z[p_id]);

					if (pixelID >= maxPixelID)
						continue;

					if (instID < 0 || primID < 0)
					{
						m_Accumulator[pixelID] += vec4(1, 0, 0, 0.0f);
						continue;
					}

					m_Accumulator[pixelID] += vec4(direction, 0.0f);
				}
			}
		});
	}
#else
	for (int j = 0; j < threads; j++)
	{
		const int t_id = j;
		handles[j] = std::async([this, packetsPerThread, nr_packets, maxPixelID, probe_id, t_id]() {
			const int start = static_cast<int>(t_id * packetsPerThread);
			const int end = static_cast<int>((t_id + 1) * packetsPerThread);

			for (int i = start; i < end; i++)
			{
				auto &packet = m_Packets[i];

				float bary_x[4];
				float bary_y[4];

#if PACKET_TRAVERSAL
				topLevelBVH.intersect4(packet.origin_x, packet.origin_y, packet.origin_z, packet.direction_x,
									   packet.direction_y, packet.direction_z, packet.t, bary_x, bary_y, packet.primID,
									   packet.instID, 1e-5f);
#else
				for (int i = 0; i < 4; i++)
				{
					const vec3 origin = vec3(packet.origin_x[i], packet.origin_y[i], packet.origin_z[i]);
					const vec3 direction = vec3(packet.direction_x[i], packet.direction_y[i], packet.direction_z[i]);

					topLevelBVH.intersect(origin, direction, &packet.t[i], &packet.primID[i], &packet.instID[i]);
				}
#endif

				for (int packet_id = 0, s = TILE_WIDTH * TILE_HEIGHT; packet_id < s; packet_id++)
				{
					const int pixelID = packet.pixelID[packet_id];

					if (pixelID >= maxPixelID)
						continue;

					if (pixelID == probe_id)
					{
						m_ProbedDist = packet.t[packet_id];
						m_ProbedInstance = packet.instID[packet_id];
						m_ProbedTriangle = packet.primID[packet_id];
					}

					int primID = packet.primID[packet_id];
					int instID = packet.instID[packet_id];
					float t = packet.t[packet_id];
					vec3 origin =
						vec3(packet.origin_x[packet_id], packet.origin_y[packet_id], packet.origin_z[packet_id]);
					vec3 direction = vec3(packet.direction_x[packet_id], packet.direction_y[packet_id],
										  packet.direction_z[packet_id]);
					vec3 throughput = vec3(1.0f);
					vec2 bary = vec2(bary_x[packet_id], bary_y[packet_id]);
					for (int pl = 0; pl < MAX_PATH_LENGTH; pl++)
					{
						if (instID < 0 || primID < 0)
						{
							const vec2 uv =
								vec2(0.5f * (1.0f + glm::atan(direction.x, -direction.z) * glm::one_over_pi<float>()),
									 glm::acos(direction.y) * glm::one_over_pi<float>());
							const uvec2 pUv = uvec2(uv.x * float(m_SkyboxWidth - 1), uv.y *float(m_SkyboxHeight - 1));
							m_Accumulator[pixelID] += vec4(throughput * m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
							break;
						}

						const vec3 p = origin + direction * t;
						const Triangle &tri = topLevelBVH.get_triangle(instID, primID);
						const auto &material = m_Materials[tri.material];

						ShadingData shadingData{};
						shadingData.color = material.getColor();

						if (any(greaterThan(shadingData.color, vec3(1.0f))))
						{
							m_Accumulator[pixelID] += vec4(throughput * shadingData.color, 0.0f);
							break;
						}

						const simd::matrix4 &matrix = topLevelBVH.get_instance_matrix(instID);
						const simd::matrix4 &normal_matrix = topLevelBVH.get_normal_matrix(instID);
						vec3 N = normalize(normal_matrix * vec3(tri.Nx, tri.Ny, tri.Nz));

						const float u = bary.x;
						const float v = bary.y;
						const float w = 1.0f - u - v;

						vec3 iN = normalize(normal_matrix * vec3(u * tri.vN0 + v * tri.vN1 + w * tri.vN2));

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
							tu = u * tri.u0 + v * tri.u1 + w * tri.u2;
							tv = u * tri.v0 + v * tri.v1 + w * tri.v2;
						}

						if (material.hasFlag(HasDiffuseMap))
						{
							const float t_u = (tu + material.uoffs0) * material.uscale0;
							const float t_v = (tv + material.voffs0) * material.vscale0;

							float tx = glm::mod(t_u, 1.0f);
							float ty = glm::mod(t_v, 1.0f);

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

						vec3 R;
						float pdf;
						const vec3 bsdf = SampleBSDF(shadingData, iN, N, T, B, -direction, t, bool(flip < 0), R, pdf,
													 m_RNGs[t_id].Rand(), m_RNGs[t_id].Rand(), m_RNGs[t_id].Rand(),
													 m_RNGs[t_id].Rand());

						throughput = throughput * bsdf * glm::abs(glm::dot(iN, R)) * (1.0f / pdf);
						if (any(isnan(throughput)) || any(lessThan(throughput, vec3(0))))
							break;

						origin = SafeOrigin(p, R, N, 1e-6f);
						direction = R;

						t = 1e34f;
						instID = -1;
						primID = -1;
						topLevelBVH.intersect(origin, direction, &t, &primID, &instID, &bary, 1e-6f);
					}
				}
			}
		});
	}
#endif

	m_Samples++;
	for (int i = 0; i < threads; i++)
		handles[i].get();

	const float scale = 1.0f / float(m_Samples);
	for (int i = 0; i < threads; i++)
	{
		handles[i] = m_Pool.push([this, threads, scale](int tId) {
			for (int y = tId; y < m_Height; y += threads)
			{
				const int y_offset = y * m_Width;
				for (int x = 0; x < m_Width; x++)
				{
					const int pixel_id = y_offset + x;
					m_Pixels[pixel_id] = m_Accumulator[pixel_id] * scale;
				}
			}
		});
	}

	for (int i = 0; i < threads; i++)
		handles[i].get();

	m_Stats.primaryTime = timer.elapsed();

	glUnmapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	CheckGL();
	glBindTexture(GL_TEXTURE_2D, m_TargetID);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, m_Width, m_Height, GL_RGBA, GL_FLOAT, nullptr);
	CheckGL();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
	CheckGL();
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
			m_Meshes.emplace_back();
	}

	m_Meshes[index].set_geometry(mesh);
}

void Context::set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
{
	topLevelBVH.set_instance(i, transform, &m_Meshes[meshIdx], m_Meshes[meshIdx].mbvh->get_aabb());
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

void Context::update() { topLevelBVH.construct_bvh(); }

void Context::set_probe_index(glm::uvec2 probePos) { m_ProbePos = probePos; }

rfw::RenderStats Context::get_stats() const { return m_Stats; }
