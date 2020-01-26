#include "PCH.h"

#define PACKET_TRAVERSAL 0

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() { glDeleteBuffers(1, &m_PboID); }

std::vector<rfw::RenderTarget> Context::get_supported_targets() const { return {OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
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
	constexpr int TILE_WIDTH = 2;
	constexpr int TILE_HEIGHT = 2;
#elif PACKET_WIDTH == 8
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 2;
#endif

	const int wTiles = (m_Width + (TILE_WIDTH - (m_Width % TILE_WIDTH))) / TILE_WIDTH;
	const int hTiles = (m_Height + (TILE_HEIGHT - (m_Height % TILE_HEIGHT))) / TILE_HEIGHT;
	const int threads = static_cast<int>(m_Pool.size());

	m_Packets.resize(wTiles * hTiles);

	std::vector<std::future<void>> handles(threads);

	for (int i = 0, s = threads; i < s; i++)
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

					const int x4[4] = {x, x + 1, x, x + 1};
					const int y4[4] = {y, y, y + 1, y + 1};
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
	const auto packetsPerThread = m_Packets.size() / threads;

	for (int j = 0, s = static_cast<int>(threads); j < s; j++)
	{
		handles[j] = m_Pool.push([&](int t_id) {
			const int start = static_cast<int>(t_id * packetsPerThread);
			const int end = static_cast<int>((t_id + 1) * packetsPerThread);

			for (int i = start; i < end; i++)
			{
				auto &packet = m_Packets[i];

#if PACKET_TRAVERSAL
				if (topLevelBVH.intersect4(packet.origin_x, packet.origin_y, packet.origin_z, packet.direction_x,
										   packet.direction_y, packet.direction_z, packet.t, packet.primID,
										   packet.instID, 1e-5f) == 0)
				{
					for (int i = 0; i < 4; i++)
					{
						if (packet.pixelID[i] >= maxPixelID)
							continue;
						const vec2 uv = vec2(0.5f * (1.0f + glm::atan(packet.direction_x[i], -packet.direction_z[i]) *
																glm::one_over_pi<float>()),
											 glm::acos(packet.direction_y[i]) * glm::one_over_pi<float>());
						const uvec2 pUv = uvec2(uv.x * float(m_SkyboxWidth - 1), uv.y *float(m_SkyboxHeight - 1));
						m_Accumulator[packet.pixelID[i]] += glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
					}

					continue;
				}
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

						ShadingData shadingData;
						shadingData.color = material.getColor();

						if (any(greaterThan(shadingData.color, vec3(1.0f))))
						{
							m_Accumulator[pixelID] += vec4(throughput * shadingData.color, 0.0f);
							break;
						}

						const simd::matrix4 &matrix = topLevelBVH.get_instance_matrix(instID);
						const simd::matrix4 &normal_matrix = topLevelBVH.get_normal_matrix(instID);
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

						t = 1e34f;
						instID = -1;
						primID = -1;
						topLevelBVH.intersect(origin, direction, &t, &primID, &instID, 1e-5f);
					}
				}
			}
		});
	}

	m_Samples++;

	for (int i = 0; i < threads; i++)
		handles[i].get();

	const float scale = 1.0f / float(m_Samples);
	for (int i = 0, s = threads; i < s; i++)
	{
		handles[i] = m_Pool.push([this, &threads, &scale](int tId) {
			for (int y = tId; y < m_Height; y += threads)
			{
				const uint y_offset = y * m_Width;
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
	m_SkyboxWidth = width;
	m_SkyboxHeight = height;
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
