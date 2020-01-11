#pragma once

#include <RenderContext.h>
#include <ContextExport.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>
#include <utils/ThreadPool.h>
#include <utils/Xor128.h>

#include <GL/glew.h>

#include "Mesh.h"
#include "BVH/TopLevelBVH.h"

#define PACKET_WIDTH 4

namespace rfw
{

class Context : public RenderContext
{
	~Context() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override;

	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void setTextures(const std::vector<rfw::TextureData> &textures) override;
	void setMesh(size_t index, const rfw::Mesh &mesh) override;
	void setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
				   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights) override;
	void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	rfw::RenderStats getStats() const override;

  private:
	rfw::RenderStats m_Stats;
	LightCount m_LightCount;
	std::vector<PointLight> m_PointLights;
	std::vector<AreaLight> m_AreaLights;
	std::vector<DirectionalLight> m_DirectionalLights;
	std::vector<SpotLight> m_SpotLights;
	std::vector<Material> m_Materials;
	std::vector<TextureData> m_Textures;

	utils::ThreadPool m_Pool = {};
	std::vector<std::future<void>> m_Handles;
	std::vector<utils::Xor128> m_RNGs;

#if PACKET_WIDTH == 4
	std::vector<cpurt::RayPacket4> m_Packets;
#elif PACKET_WIDTH == 8
	std::vector<cpurt::RayPacket8> m_Packets;
#endif

	TopLevelBVH topLevelBVH;
	std::vector<CPUMesh> m_Meshes;

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
};

} // namespace rfw