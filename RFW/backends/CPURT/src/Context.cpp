#include "PCH.h"

using namespace rfw;

void createTangentSpace(const vec3 N, vec3 &T, vec3 &B)
{
	const float s = sign(N.z);
	const float a = -1.0f / (s + N.z);
	const float b = N.x * N.y * a;
	T = vec3(1.0f + s * N.x * N.x * a, s * b, -s * N.x);
	B = vec3(b, s + N.y * N.y * a, -N.y);
}

vec3 tangentToWorld(const vec3 s, const vec3 N, const vec3 T, const vec3 B) { return T * s.x + B * s.y + N * s.z; }

vec3 worldToTangent(const vec3 s, const vec3 N, const vec3 T, const vec3 B)
{
	return vec3(dot(T, s), dot(B, s), dot(N, s));
}

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context() { glDeleteBuffers(1, &m_PboID); }

std::vector<rfw::RenderTarget> Context::get_supported_targets() const { return {OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		m_Handles.resize(m_Pool.size());
		m_RNGs.resize(m_Pool.size());
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::xor128(i + 1);

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
		for (int i = 0, s = static_cast<int>(m_Pool.size()); i < s; i++)
			m_RNGs[i] = utils::xor128(i + 1);
	}

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	if (m_Pixels == nullptr)
		throw std::runtime_error("Could not obtain pointer to pixel buffer.");

	const auto camParams = cpurt::Ray::CameraParams(camera.get_view(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	const auto timer = utils::timer();

#if PACKET_WIDTH == 4
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 1;
#elif PACKET_WIDTH == 8
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 2;
#endif

	const int probe_id = static_cast<int>(m_ProbePos.y * m_Width + m_ProbePos.x);
	const int maxPixelID = m_Width * m_Height;

	tbb::parallel_for(
		tbb::blocked_range2d<int, int>(0, m_Height, 0, m_Width / 4), [&](const tbb::blocked_range2d<int, int> &r) {
			const auto rows = r.rows();
			const auto cols = r.cols();

			for (int y = rows.begin(); y < rows.end(); y++)
			{
				for (int x_4 = cols.begin(); x_4 < cols.end(); x_4++)
				{
					const int x = x_4 * 4;

					const int x4[4] = {x, x + 1, x + 2, x + 3};
					const int y4[4] = {y, y, y, y};

					auto packet = cpurt::Ray::generate_ray4(camParams, x4, y4, &m_RNGs[x_4 % m_Pool.size()]);

					if (m_packet_traversal)
					{
						if (topLevelBVH.intersect4(packet.origin_x, packet.origin_y, packet.origin_z,
												   packet.direction_x, packet.direction_y, packet.direction_z, packet.t,
												   packet.primID, packet.instID, 1e-5f) == 0)
						{
							for (int instance = 0; instance < 4; instance++)
							{
								if (packet.pixelID[instance] >= maxPixelID)
									continue;
								const vec2 uv = vec2(
									0.5f * (1.0f + atan(packet.direction_x[instance], -packet.direction_z[instance]) *
													   glm::one_over_pi<float>()),
									acos(packet.direction_y[instance]) * glm::one_over_pi<float>());
								const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1),
														uv.y * static_cast<float>(m_SkyboxHeight - 1));
								m_Pixels[packet.pixelID[instance]] =
									glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
							}

							continue;
						}
					}
					else
					{
						for (int instance = 0; instance < 4; instance++)
						{
							const vec3 origin =
								vec3(packet.origin_x[instance], packet.origin_y[instance], packet.origin_z[instance]);
							const vec3 direction = vec3(packet.direction_x[instance], packet.direction_y[instance],
														packet.direction_z[instance]);

							topLevelBVH.intersect(origin, direction, &packet.t[instance], &packet.primID[instance],
												  &packet.instID[instance]);
						}
					}

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
						const vec3 origin =
							vec3(packet.origin_x[packet_id], packet.origin_y[packet_id], packet.origin_z[packet_id]);
						const vec3 direction = vec3(packet.direction_x[packet_id], packet.direction_y[packet_id],
													packet.direction_z[packet_id]);

						if (packet.instID[packet_id] < 0)
						{
							const vec2 uv =
								vec2(0.5f * (1.0f + atan(direction.x, -direction.z) * glm::one_over_pi<float>()),
									 acos(direction.y) * glm::one_over_pi<float>());
							const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1),
													uv.y * static_cast<float>(m_SkyboxHeight - 1));
							m_Pixels[pixelID] = glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
							continue;
						}

						if (packet.primID[packet_id] < 0 || packet.instID[packet_id] < 0)
						{
							m_Pixels[pixelID] = glm::vec4(1, 0, 0, 0);
							continue;
						}

						const vec3 p = origin + direction * t;
						const Triangle &tri =
							topLevelBVH.get_triangle(packet.instID[packet_id], packet.primID[packet_id]);
						const simd::matrix4 &matrix = topLevelBVH.get_instance_matrix(packet.instID[packet_id]);
						const simd::matrix4 &normal_matrix = topLevelBVH.get_normal_matrix(packet.instID[packet_id]);

						const auto &material = m_Materials[tri.material];
						const auto shading_data = retrieve_material(tri, material, p, matrix, normal_matrix);
						if (any(greaterThan(shading_data.color, vec3(1))))
						{
							m_Pixels[pixelID] = vec4(shading_data.color, 1.0f);
							continue;
						}

						vec3 contrib = vec3(0.1f);
						for (const auto &l : m_AreaLights)
						{
							vec3 L = l.position - p;
							const float sq_dist = dot(L, L);
							const float dist = sqrt(sq_dist);
							L = L / dist;
							const float NdotL = dot(shading_data.iN, L);
							const float LNdotL = -dot(l.normal, L);

							if (NdotL <= 0 || LNdotL <= 0)
								continue;

							if (!topLevelBVH.is_occluded(p, L, dist - 2.0f * 1e-5f, 1e-4f))
								contrib += l.radiance * l.area / sq_dist * NdotL * LNdotL;
						}

						for (const auto &l : m_PointLights)
						{
							vec3 L = l.position - p;
							const float sq_dist = dot(L, L);
							const float dist = sqrt(sq_dist);
							L = L / dist;
							const float NdotL = dot(shading_data.iN, L);
							if (NdotL <= 0)
								continue;

							if (!topLevelBVH.is_occluded(p, L, dist - 2.0f * 1e-5f, 1e-4f))
								contrib += l.radiance / sq_dist * NdotL;
						}

						// for (const auto &l : m_DirectionalLights)
						//{
						//}

						// for (const auto &l : m_SpotLights)
						//{
						//}

						m_Pixels[pixelID] = vec4(shading_data.color * contrib, 1.0f);
					}
				}
			}
		});

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

rfw::AvailableRenderSettings Context::get_settings() const
{
	auto settings = rfw::AvailableRenderSettings();
	settings.settingKeys = {"packet_traversal"};
	settings.settingValues = {{"1", "0"}};
	return settings;
}

void Context::set_setting(const rfw::RenderSetting &setting)
{
	if (setting.name == "packet_traversal")
		m_packet_traversal = setting.value == "1" ? true : false;
}

void Context::update() { topLevelBVH.construct_bvh(); }

void Context::set_probe_index(glm::uvec2 probePos) { m_ProbePos = probePos; }

rfw::RenderStats Context::get_stats() const { return m_Stats; }

Context::ShadingData Context::retrieve_material(const Triangle &tri, const Material &material, const glm::vec3 &p,
												const simd::matrix4 &matrix, const simd::matrix4 &normal_matrix) const
{
	ShadingData data{};
	data.N = normalize(vec3((normal_matrix * simd::vector4(tri.Nx, tri.Ny, tri.Nz, 0.0f)).vec));

	const simd::vector4 vertex0_4 = matrix * simd::vector4(tri.vertex0.x, tri.vertex0.y, tri.vertex0.z, 1.0f);
	const simd::vector4 vertex1_4 = matrix * simd::vector4(tri.vertex1.x, tri.vertex1.y, tri.vertex1.z, 1.0f);
	const simd::vector4 vertex2_4 = matrix * simd::vector4(tri.vertex2.x, tri.vertex2.y, tri.vertex2.z, 1.0f);

	const vec3 vertex0 = vec3(vertex0_4.vec);
	const vec3 vertex1 = vec3(vertex1_4.vec);
	const vec3 vertex2 = vec3(vertex2_4.vec);

	const vec3 bary = triangle::getBaryCoords(p, data.N, vertex0, vertex1, vertex2);

	const vec3 iN = bary.x * tri.vN0 + bary.y * tri.vN1 + bary.z * tri.vN2;
	const simd::vector4 vN_4 = normal_matrix * simd::vector4(iN.x, iN.y, iN.z, 0.0f);
	data.iN = normalize(vec3(vN_4.vec));

	createTangentSpace(iN, data.T, data.B);

	data.color = material.getColor();

	float tu = 0.0f, tv = 0.0f;
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

		float tx = fmod(u, 1.0f);
		float ty = fmod(v, 1.0f);

		if (tx < 0.f)
			tx = 1.f + tx;
		if (ty < 0.f)
			ty = 1.f + ty;

		const auto &tex = m_Textures[material.texaddr0];

		const uint ix = uint(tx * static_cast<float>(tex.width - 1));
		const uint iy = uint(ty * static_cast<float>(tex.height - 1));
		const auto texel_id = static_cast<int>(iy * tex.width + ix);

		switch (tex.type)
		{
		case (TextureData::FLOAT4):
		{
			data.color = data.color * vec3(reinterpret_cast<vec4 *>(tex.data)[texel_id]);
		}
		case (TextureData::UINT):
		{
			// RGBA
			const uint texel_color = reinterpret_cast<uint *>(tex.data)[texel_id];
			constexpr float texture_scale = 1.0f / 256.0f;
			data.color = data.color * texture_scale *
						 vec3(texel_color & 0xFFu, (texel_color >> 8u) & 0xFFu, (texel_color >> 16u) & 0xFFu);
		}
		}
	}

	return data;
}
