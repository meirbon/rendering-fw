#include "Context.h"

#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>
#include <utils/Timer.h>
#include <utils/gl/CheckGL.h>

#include <utils/Xor128.h>

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

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { utils::logger::err("Window not supported (yet)."); }

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
		{
			utils::logger::err("Could not init GLEW.");
		}
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
	glFinish();
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, m_PboID);
	m_Pixels = static_cast<glm::vec4 *>(glMapBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, GL_WRITE_ONLY_ARB));
	assert(m_Pixels);
	CheckGL();

	const auto camParams = Ray::CameraParams(camera.get_view(), 0, 1e-5f, m_Width, m_Height);

	m_Stats.clear();
	m_Stats.primaryCount = m_Width * m_Height;

	auto timer = utils::Timer();

	const int maxPixelID = m_Width * m_Height;
	const __m128i maxPixelID4 = _mm_set1_epi32(maxPixelID - 1);
	const __m256i maxPixelID8 = _mm256_set1_epi32(maxPixelID - 1);
	const int probe_id = m_ProbePos.y * m_Width + m_ProbePos.x;

#if PACKET_WIDTH == 4
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 1;
	const int x_offs[4] = {0, 1, 0, 1};
	const int y_offs[4] = {0, 0, 1, 1};
#elif PACKET_WIDTH == 8
	constexpr int TILE_WIDTH = 4;
	constexpr int TILE_HEIGHT = 2;

	const int x_offs[8] = {0, 1, 2, 3, 0, 1, 2, 3};
	const int y_offs[8] = {0, 0, 0, 0, 1, 1, 1, 1};
#endif

	tbb::parallel_for(
		tbb::blocked_range2d<int, int>(0, m_Height / TILE_HEIGHT, 0, m_Width / TILE_WIDTH),
		[&](const tbb::blocked_range2d<int, int> &r) {
			const auto rows = r.rows();
			const auto cols = r.cols();

			RTCIntersectContext context, shadow_context;
			rtcInitIntersectContext(&context);
			rtcInitIntersectContext(&shadow_context);
			context.flags = RTC_INTERSECT_CONTEXT_FLAG_COHERENT;
			shadow_context.flags = RTC_INTERSECT_CONTEXT_FLAG_INCOHERENT;
			int valid[PACKET_WIDTH];

			for (int y_l = rows.begin(); y_l < rows.end(); y_l++)
			{
				for (int x_l = cols.begin(); x_l < cols.end(); x_l++)
				{
					memset(valid, -1, sizeof(valid));

					const int x = x_l * TILE_WIDTH;
					const int y = y_l * TILE_HEIGHT;

					int xs[PACKET_WIDTH];
					int ys[PACKET_WIDTH];

					for (int i = 0; i < PACKET_WIDTH; i++)
					{
						xs[i] = x + x_offs[i];
						ys[i] = y + y_offs[i];
					}

#if PACKET_WIDTH == 4
					auto packet = Ray::GenerateRay4(camParams, xs, ys, &m_Rng);
#elif PACKET_WIDTH == 8
					auto packet = Ray::GenerateRay8(camParams, xs, ys, &m_Rng);
#endif

#if PACKET_WIDTH == 4
					rtcIntersect4(valid, m_Scene, &context, &packet);
#elif PACKET_WIDTH == 8
					rtcIntersect8(valid, m_Scene, &context, &packet);
#endif
					for (int j = 0; j < PACKET_WIDTH; j++)
					{
						const vec3 origin = vec3(packet.ray.org_x[j], packet.ray.org_y[j], packet.ray.org_z[j]);
						const vec3 direction = vec3(packet.ray.dir_x[j], packet.ray.dir_y[j], packet.ray.dir_z[j]);

						const int &pixel_id = packet.ray.id[j];
						if (pixel_id >= maxPixelID)
							continue;
						if (packet.hit.geomID[j] == RTC_INVALID_GEOMETRY_ID)
						{
							const vec2 uv =
								vec2(0.5f * (1.0f + atan(direction.x, -direction.z) * glm::one_over_pi<float>()),
									 acos(direction.y) * glm::one_over_pi<float>());
							const uvec2 pUv = uvec2(uv.x * static_cast<float>(m_SkyboxWidth - 1),
													uv.y * static_cast<float>(m_SkyboxHeight - 1));
							m_Pixels[pixel_id] = glm::vec4(m_Skybox[pUv.y * m_SkyboxWidth + pUv.x], 0.0f);
							continue;
						}

						const int &instID = packet.hit.instID[0][j];
						const int &primID = packet.hit.primID[j];

						if (pixel_id == probe_id)
						{
							m_ProbedDist = packet.ray.tfar[j];
							m_ProbedInstance = instID;
							m_ProbedTriangle = primID;
						}

						const simd::matrix4 &normal_matrix = m_InverseMatrices[instID];
						const Triangle &tri = m_Meshes[m_InstanceMesh[instID]].triangles[primID];
						const vec3 bary =
							vec3(1.0f - packet.hit.u[j] - packet.hit.v[j], packet.hit.u[j], packet.hit.v[j]);
						const vec3 p = origin + direction * packet.ray.tfar[j];

						const auto &material = m_Materials[tri.material];
						const auto shading_data = retrieve_material(tri, material, p, bary, normal_matrix);
						if (any(greaterThan(shading_data.color, vec3(1))))
						{
							m_Pixels[pixel_id] = vec4(shading_data.color, 1.0f);
							continue;
						}

						RTCRay ray{};
						ray.tnear = 1e-4f;
						ray.org_x = p.x;
						ray.org_y = p.y;
						ray.org_z = p.z;

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

							ray.tfar = dist;
							ray.flags = 0;
							ray.dir_x = L.x;
							ray.dir_y = L.y;
							ray.dir_z = L.z;

							rtcOccluded1(m_Scene, &shadow_context, &ray);
							if (ray.tfar > 0)
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

							ray.tfar = dist;
							ray.flags = 0;
							ray.dir_x = L.x;
							ray.dir_y = L.y;
							ray.dir_z = L.z;

							rtcOccluded1(m_Scene, &shadow_context, &ray);
							if (ray.tfar > 0)
								contrib += l.radiance / sq_dist * NdotL;
						}

						// for (const auto &l : m_DirectionalLights)
						//{
						//}

						// for (const auto &l : m_SpotLights)
						//{
						//}

						m_Pixels[pixel_id] = vec4(shading_data.color * contrib, 1.0f);
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

	m_InstanceMesh[i] = uint(meshIdx);
	m_InstanceMatrices[i] = transform;
	m_InverseMatrices[i] = mat4(inverse_transform);
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

Context::ShadingData Context::retrieve_material(const Triangle &tri, const Material &material, const glm::vec3 &p,
												const glm::vec3 bary, const simd::matrix4 &normal_matrix) const
{
	ShadingData data{};
	data.N = normalize(vec3((normal_matrix * simd::vector4(tri.Nx, tri.Ny, tri.Nz, 0.0f)).vec));

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
