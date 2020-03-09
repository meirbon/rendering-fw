#pragma once

#include "PCH.h"

#define PACKET_WIDTH 8

namespace rfw
{

class Context : public RenderContext
{
	~Context() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> get_supported_targets() const override;

	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void render_frame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void set_materials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void set_textures(const std::vector<rfw::TextureData> &textures) override;
	void set_mesh(size_t index, const rfw::Mesh &mesh) override;
	void set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
				   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights) override;
	void get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	rfw::AvailableRenderSettings get_settings() const override;
	void set_setting(const rfw::RenderSetting &setting) override;
	void update() override;
	void set_probe_index(glm::uvec2 probePos) override;
	rfw::RenderStats get_stats() const override;

  private:
	struct ShadingData
	{
		glm::vec3 color;
		glm::vec3 N;
		glm::vec3 iN;
		glm::vec3 T;
		glm::vec3 B;
	};
	
	ShadingData retrieve_material(const Triangle &tri, const Material &material, const glm::vec3 &p,
								  const glm::vec3 bary, const simd::matrix4 &normal_matrix) const;
	
	rfw::RenderStats m_Stats;
	LightCount m_LightCount;
	std::vector<PointLight> m_PointLights;
	std::vector<AreaLight> m_AreaLights;
	std::vector<DirectionalLight> m_DirectionalLights;
	std::vector<SpotLight> m_SpotLights;
	std::vector<Material> m_Materials;
	std::vector<TextureData> m_Textures;

	std::vector<CPUMesh> m_Meshes;

	RTCDevice m_Device;
	RTCScene m_Scene;

	std::vector<bool> m_MeshChanged;
	std::vector<uint> m_Instances;
	std::vector<uint> m_InstanceMesh;
	std::vector<simd::matrix4> m_InstanceMatrices;
	std::vector<simd::matrix4> m_InverseMatrices;

	rfw::utils::ThreadPool m_Pool = {};
	int m_SkyboxWidth = 0, m_SkyboxHeight = 0;
	std::vector<glm::vec3> m_Skybox = {glm::vec3(0)};
	glm::vec4 *m_Pixels = nullptr;
	GLuint m_TargetID = 0, m_PboID = 0;
	int m_Width, m_Height;
	glm::uvec2 m_ProbePos = glm::uvec2(0);
	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedTriangle = 0;
	float m_ProbedDist = -1.0f;
	bool m_InitializedGlew = false;

	utils::Xor128 m_Rng = {};

#if PACKET_WIDTH == 4
	const int TILE_WIDTH = 2;
	const int TILE_HEIGHT = 2;

	std::vector<RTCRayHit4> m_Packets;
#elif PACKET_WIDTH == 8
	const int TILE_WIDTH = 4;
	const int TILE_HEIGHT = 2;

	std::vector<RTCRayHit8> m_Packets;
#elif PACKET_WIDTH == 16
	const int TILE_WIDTH = 4;
	const int TILE_HEIGHT = 4;

	std::vector<RTCRayHit16> m_Packets;
#endif
};

} // namespace rfw
