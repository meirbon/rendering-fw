#include "PCH.h"

#define PACKET_TRAVERSAL 0

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() { glDeleteBuffers(1, &m_PboID); }

std::vector<rfw::RenderTarget> Context::getSupportedTargets() const { return {OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		m_Handles.resize(m_Pool.size());
		m_RNGs.resize(m_Pool.size());
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::Xor128(i + 1);

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
	if (status == Reset)
	{
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::Xor128(i + 1);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);

	const auto camParams = cpurt::Ray::CameraParams(camera.getView(), 0, 1e-5f, m_Width, m_Height);

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
			m_Packets[tile_id] = cpurt::Ray::generateRay4(camParams, x4, y4, &m_RNGs[tile_y % m_RNGs.size()]);
#elif PACKET_WIDTH == 8
			const int y = tile_y * TILE_HEIGHT;
			const int x = tile_x * TILE_WIDTH;
			const int tile_id = tile_y * wTiles + tile_x;

			const int x8[8] = {x, x + 1, x + 2, x + 3, x, x + 1, x + 2, x + 3};
			const int y8[8] = {y, y, y, y, y + 1, y + 1, y + 1, y + 1};
			m_Packets[tile_id] = Ray::GenerateRay8(camParams, x8, y8, &m_RNGs[tile_y % m_RNGs.size()]);
#endif
		}
	});

	const int probe_id = m_ProbePos.y * m_Width + m_ProbePos.x;
	const int maxPixelID = m_Width * m_Height;
	const auto threads = m_Pool.size();
	const auto packetsPerThread = m_Packets.size() / threads;
	std::vector<std::future<void>> handles;
	handles.reserve(threads);

#if 1
	for (size_t i = 0; i < threads; i++)
	{
		handles.push_back(m_Pool.push([&](int t_id) {
			const int start = static_cast<int>(t_id * packetsPerThread);
			const int end = static_cast<int>((t_id + 1) * packetsPerThread);

#if PACKET_TRAVERSAL
			for (int i = start; i < end; i++)
			{
				auto &packet = m_Packets[i];
				topLevelBVH.intersect(packet, 1e-5f);

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

					const float t = packet.t[packet_id];
					const vec3 origin = vec3(packet.origin_x[packet_id], packet.origin_y[packet_id], packet.origin_z[packet_id]);
					const vec3 direction = vec3(packet.direction_x[packet_id], packet.direction_y[packet_id], packet.direction_z[packet_id]);

					if (packet.instID[packet_id] < 0)
					{
						const vec2 uv =
							vec2(0.5f * (1.0f + atan(direction.x, -direction.z) * glm::one_over_pi<float>()), acos(direction.y) * glm::one_over_pi<float>());
						const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
						m_Pixels[pixelID] = glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
						continue;
					}

					const vec3 p = origin + direction * t;
					const Triangle &tri = topLevelBVH.get_triangle(packet.instID[packet_id], packet.primID[packet_id]);
					const simd::matrix4 &matrix = topLevelBVH.get_instance_matrix(packet.instID[packet_id]);
					const simd::matrix4 &normal_matrix = topLevelBVH.get_normal_matrix(packet.instID[packet_id]);

					static const __m128i normal_mask = _mm_set_epi32(0, ~0, ~0, ~0);

					vec3 N;
					_mm_maskstore_ps(value_ptr(N), normal_mask, glm_mat4_mul_vec4(normal_matrix.cols, _mm_setr_ps(tri.Nx, tri.Ny, tri.Nz, 0.0f)));
					N = normalize(N);

					const __m128 vertex0_4 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(tri.vertex0.x, tri.vertex0.y, tri.vertex0.z, 1.0f));
					const __m128 vertex1_4 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(tri.vertex1.x, tri.vertex1.y, tri.vertex1.z, 1.0f));
					const __m128 vertex2_4 = glm_mat4_mul_vec4(matrix.cols, _mm_setr_ps(tri.vertex2.x, tri.vertex2.y, tri.vertex2.z, 1.0f));

					const vec3 vertex0 = make_vec3(reinterpret_cast<const float *>(&vertex0_4));
					const vec3 vertex1 = make_vec3(reinterpret_cast<const float *>(&vertex1_4));
					const vec3 vertex2 = make_vec3(reinterpret_cast<const float *>(&vertex2_4));

					const vec3 bary = triangle::getBaryCoords(p, N, vertex0, vertex1, vertex2);

					vec3 iN = bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2;
					const __m128 vN_4 = glm_mat4_mul_vec4(normal_matrix.cols, _mm_setr_ps(iN.x, iN.y, iN.z, 0.0f));
					_mm_maskstore_ps(value_ptr(iN), normal_mask, vN_4);
					iN = normalize(iN);

					const auto &material = m_Materials[tri.material];
					auto color = material.getColor();

					float tu, tv;
					if (material.hasFlag(HasDiffuseMap) || material.hasFlag(HasNormalMap) || material.hasFlag(HasRoughnessMap) ||
						material.hasFlag(HasAlphaMap) || material.hasFlag(HasSpecularityMap))
					{
						tu = bary.x * tri.u0 + bary.y * tri.u1 + bary.z * tri.u2;
						tv = bary.x * tri.v0 + bary.y * tri.v1 + bary.z * tri.v2;
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

					m_Pixels[pixelID] = vec4(color, 1.0f);
				}
			}
#else
			for (int i = start; i < end; i++)
			{
				auto &packet = m_Packets[i];
				for (int packet_id = 0, s = TILE_WIDTH * TILE_HEIGHT; packet_id < s; packet_id++)
				{
					std::optional<Triangle> result;

					const vec3 origin = vec3(packet.origin_x[packet_id], packet.origin_y[packet_id], packet.origin_z[packet_id]);
					const vec3 direction = vec3(packet.direction_x[packet_id], packet.direction_y[packet_id], packet.direction_z[packet_id]);
					float t = 1e34f;
					int primID;
					const int pixelID = packet.pixelID[packet_id];
					if (pixelID >= maxPixelID)
						break;

					unsigned int instID = 0;
					result = topLevelBVH.intersect(origin, direction, &t, &primID, 1e-5f, instID);
					if (result.has_value() && pixelID == probe_id)
					{
						m_ProbedDist = t;
						m_ProbedInstance = instID;
						m_ProbedTriangle = primID;
					}
					else if (!result.has_value())
					{
						const float inv_pi = glm::one_over_pi<float>();

						const vec2 uv =
							vec2(0.5f * (1.0f + atan(direction.x, -direction.z) * glm::one_over_pi<float>()), acos(direction.y) * glm::one_over_pi<float>());
						const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
						m_Pixels[pixelID] = vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
						continue;
					}

					const auto &tri = result.value();
					const vec3 N = vec3(tri.Nx, tri.Ny, tri.Nz);
					const vec3 p = origin + direction * t;
					const vec3 bary = triangle::getBaryCoords(p, N, tri.vertex0, tri.vertex1, tri.vertex2);
					const vec3 iN = normalize(bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2);
					const auto &material = m_Materials[tri.material];
					auto color = material.getColor();

					float tu, tv;
					if (material.hasFlag(HasDiffuseMap) || material.hasFlag(HasNormalMap) || material.hasFlag(HasRoughnessMap) ||
						material.hasFlag(HasAlphaMap) || material.hasFlag(HasSpecularityMap))
					{
						tu = bary.x * tri.u0 + bary.y * tri.u1 + bary.z * tri.u2;
						tv = bary.x * tri.v0 + bary.y * tri.v1 + bary.z * tri.v2;
					}

					if (material.hasFlag(HasDiffuseMap))
					{
						const float u = (tu + material.uoffs0) * material.uscale0;
						const float v = (tv + material.voffs0) * material.vscale0;

						float x = fmod(u, 1.0f);
						float y = fmod(v, 1.0f);

						if (x < 0.0f)
							x = 1.0f + x;
						if (y < 0.0f)
							y = 1.0f + y;

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

					m_Pixels[pixelID] = vec4(color, 1.0f);
				}
			}
#endif
		}));
	}

#else
	for (int i = 0; i < s; i++)
	{
		handles.push_back(m_Pool.push([&](int t_id) {
			auto &rng = m_RNGs[t_id];

			int y = t_id;
			while (y < m_Height)
			{
				const int yOffset = y * m_Width;

				for (int x = 0; x < m_Width; x++)
				{
					const int pixelIdx = yOffset + x;

					auto ray = cpurt::Ray::generateFromView(camParams, x, y, rng.Rand(), rng.Rand(), rng.Rand(), rng.Rand());
					std::optional<Triangle> result;

					unsigned int instID = 0;
					if (x == m_ProbePos.x && y == m_ProbePos.y)
					{
						result = topLevelBVH.intersect(ray, 1e-5f, instID);
						m_ProbedDist = ray.t;
						m_ProbedInstance = instID;
						m_ProbedTriangle = ray.primIdx;
					}
					else
					{
						result = topLevelBVH.intersect(ray, 1e-5f, instID);
					}

					if (!result.has_value())
					{
						const vec2 uv = vec2(0.5f * (1.0f + atan(ray.direction.x, -ray.direction.z) * glm::one_over_pi<float>()),
											 acos(ray.direction.y) * glm::one_over_pi<float>());
						const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1), uv.y * static_cast<float>(m_SkyboxHeight - 1));
						m_Pixels[pixelIdx] = glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
						continue;
					}

					const auto &tri = result.value();
					const vec3 N = vec3(tri.Nx, tri.Ny, tri.Nz);
					const vec3 p = ray.origin + ray.direction * ray.t;
					const vec3 bary = triangle::getBaryCoords(p, N, tri.vertex0, tri.vertex1, tri.vertex2);
					const vec3 iN = normalize(bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2);
					const auto &material = m_Materials[tri.material];
					auto color = material.getColor();

					float tu, tv;
					if (material.hasFlag(HasDiffuseMap) || material.hasFlag(HasNormalMap) || material.hasFlag(HasRoughnessMap) ||
						material.hasFlag(HasAlphaMap) || material.hasFlag(HasSpecularityMap))
					{
						tu = bary.x * tri.u0 + bary.y * tri.u1 + bary.z * tri.u2;
						tv = bary.x * tri.v0 + bary.y * tri.v1 + bary.z * tri.v2;
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
						const auto pixelID = static_cast<int>(iy * tex.width + ix);

						switch (tex.type)
						{
						case (TextureData::FLOAT4):
						{
							color = color * vec3(reinterpret_cast<vec4 *>(tex.data)[pixelID]);
						}
						case (TextureData::UINT):
						{
							// RGBA
							const uint texel = reinterpret_cast<uint *>(tex.data)[pixelID];
							constexpr float s = 1.0f / 256.0f;

							color = color * s * vec3(texel & 0xFFu, (texel >> 8u) & 0xFFu, (texel >> 16u) & 0xFFu);
						}
						}
					}

					m_Pixels[pixelIdx] = vec4(color, 1.0f);
				}
				y = y + s;
			}
		}));
	}
#endif

	for (auto &handle : handles)
		handle.get();

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
			m_Meshes.emplace_back();
	}

	m_Meshes[index].setGeometry(mesh);
}

void Context::setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
{
	topLevelBVH.setInstance(i, transform, &m_Meshes[meshIdx], m_Meshes[meshIdx].mbvh->get_aabb());
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

void Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const
{
	*instanceIndex = m_ProbedInstance;
	*primitiveIndex = m_ProbedTriangle;
	*distance = m_ProbedDist;
}

rfw::AvailableRenderSettings Context::getAvailableSettings() const { return {}; }

void Context::setSetting(const rfw::RenderSetting &setting) {}

void Context::update() { topLevelBVH.constructBVH(); }

void Context::setProbePos(glm::uvec2 probePos) { m_ProbePos = probePos; }

rfw::RenderStats Context::getStats() const { return m_Stats; }
