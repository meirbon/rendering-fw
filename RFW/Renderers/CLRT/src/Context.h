#pragma once

#include "PCH.h"

namespace rfw
{
class RTContext : public RenderContext
{
  public:
	RTContext();
	~RTContext() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override;

	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void setMaterials(const std::vector<rfw::DeviceMaterial> &materials,
					  const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void setTextures(const std::vector<rfw::TextureData> &textures) override;
	void setMesh(size_t index, const rfw::Mesh &mesh) override;
	void setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
				   const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
				   const rfw::DeviceDirectionalLight *directionalLights) override;
	void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	[[nodiscard]] rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	[[nodiscard]] rfw::RenderStats getStats() const override;

  private:
	void setPhase(uint phase);
	void setAccumulator();
	void setSkybox();
	void setLights();
	void setBlueNoise();
	void setCamera();
	void setPCs();
	void setOrigins();
	void setDirections();
	void setStates();
	void setCounters();
	void setMaterials();
	void setTextures();
	void setStride();
	void setPathLength(uint length);
	void setPathCount(uint pathCount);

	void resize_buffers();

	std::vector<rfw::TextureData> m_TexDescriptors;

	std::shared_ptr<cl::CLContext> m_Context;
	cl::CLBuffer<glm::vec4, cl::BufferType::TARGET> *m_Target = nullptr;

	cl::CLKernel *m_DebugKernel = nullptr;
	cl::CLKernel *m_RayGenKernel = nullptr;
	cl::CLKernel *m_IntersectKernel = nullptr;
	cl::CLKernel *m_ShadeKernel = nullptr;
	cl::CLKernel *m_FinalizeKernel = nullptr;

	cl::CLBuffer<glm::vec4> *m_Skybox = nullptr;
	cl_uint m_SkyboxWidth = 0, m_SkyboxHeight = 0;

	LightCount m_LightCount{};
	cl::CLBuffer<DeviceAreaLight> *m_AreaLights = nullptr;
	cl::CLBuffer<DevicePointLight> *m_PointLights = nullptr;
	cl::CLBuffer<DeviceSpotLight> *m_SpotLights = nullptr;
	cl::CLBuffer<DeviceDirectionalLight> *m_DirLights = nullptr;

	cl::CLBuffer<uint> *m_BlueNoise = nullptr;
	cl::CLBuffer<CLCamera> *m_Camera = nullptr;

	cl::CLBuffer<CLPotentialContribution> *m_PCs = nullptr;
	cl::CLBuffer<float4> *m_Origins = nullptr;
	cl::CLBuffer<float4> *m_Directions = nullptr;
	cl::CLBuffer<float4> *m_States = nullptr;
	cl::CLBuffer<float4> *m_Accumulator = nullptr;

	cl::CLBuffer<uint> *m_Counters = nullptr;

	cl::CLBuffer<CLMaterial> *m_Materials = nullptr;
	cl::CLBuffer<uint> *m_UintTextures = nullptr;
	cl::CLBuffer<float4> *m_FloatTextures = nullptr;

	GLuint m_TargetID{};
	GLuint m_Width{}, m_Height{};
	uint m_SampleIndex = 0;
	bool m_InitializedGlew = false;
};
} // namespace rfw
