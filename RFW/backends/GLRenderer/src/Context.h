#pragma once

#include "PCH.h"

namespace rfw
{

class Context : public RenderContext
{
	~Context() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> get_supported_targets() const override;

	void init(std::shared_ptr<rfw::utils::window> &window) override;
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
	utils::texture m_Skybox;
	glm::vec3 m_Ambient = glm::vec3(0.15f);

	void setLights(utils::shader *shader);

	std::vector<std::vector<glm::mat4>> m_Instances;
	std::vector<std::vector<glm::mat4>> m_InverseInstances;

	std::vector<int> m_InstanceGeometry;
	std::vector<glm::mat4> m_InstanceMatrices;
	std::vector<glm::mat4> m_InverseInstanceMatrices;

	std::vector<utils::texture> m_Textures;
	std::vector<GLint> m_TextureBindings;
	std::vector<DeviceMaterial> m_Materials;

	LightCount m_LightCount;
	std::vector<PointLight> m_PointLights;
	std::vector<AreaLight> m_AreaLights;
	std::vector<DirectionalLight> m_DirectionalLights;
	std::vector<SpotLight> m_SpotLights;

	utils::shader *m_CurrentShader;

	utils::shader *m_ColorShader;
	utils::shader *m_NormalShader;
	utils::shader *m_SimpleShader;
	std::vector<GLMesh *> m_Meshes;
	bool m_InitializedGlew = false;
	GLuint m_TargetID, m_FboID, m_RboID;
	GLuint m_Width, m_Height;
	rfw::RenderStats m_RenderStats;
};

} // namespace rfw
