#pragma once

#define NOMINMAX

#include <GL/glew.h>

#include <optix.h>

#include <RenderContext.h>
#include <ContextExport.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>

#include "CUDABuffer.h"
#include "OptiXCUDABuffer.h"
#include "OptiXMesh.h"

#include "SharedStructs.h"
#include "TextureInterop.h"

#include <utils/gl/GLShader.h>

class OptiXContext : public rfw::RenderContext
{
  public:
	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override { return {rfw::RenderTarget::OPENGL_TEXTURE, rfw::RenderTarget::WINDOW}; }
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
	[[nodiscard]] rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	[[nodiscard]] rfw::RenderStats getStats() const override;

  private:
	bool m_Initialized = false, m_Denoise = false, m_FirstFrame = true;
	rfw::RenderTarget m_CurrentTarget;
	rfw::RenderStats m_RenderStats;
	void resizeBuffers();
	void createTexture();

	unsigned int m_SampleIndex = 0;
	float m_GeometryEpsilon = 0.0001f;
	int m_ScrWidth, m_ScrHeight;

	std::shared_ptr<rfw::utils::Window> m_Window;
	rfw::utils::GLShader *m_Shader = nullptr;
	GLuint m_TextureID = 0;
	TextureInterop m_CUDASurface;

	optix::Context m_Context;
	optix::Group m_SceneGraph;
	optix::Acceleration m_Acceleration;
	optix::Program m_AttribProgram;
	optix::Program m_PrimaryRayProgram;
	optix::Program m_SecondaryRayProgram;
	optix::Program m_ShadowRayProgram;
	optix::Program m_ClosestHit;
	optix::Program m_ShadowMissProgram;
	optix::Material m_Material;

	CUDABuffer<glm::vec4> *m_FloatTextures = nullptr;
	CUDABuffer<uint> *m_UintTextures = nullptr;
	CUDABuffer<glm::vec3> *m_Skybox = nullptr;

	CUDABuffer<rfw::DeviceMaterial> *m_Materials = nullptr;
	std::vector<rfw::DeviceInstanceDescriptor> m_InstanceDescriptors;
	CUDABuffer<rfw::DeviceInstanceDescriptor> *m_DeviceInstanceDescriptors = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_Accumulator = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_PathStates = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_PathOrigins = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_PathDirections = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_PathThroughputs = nullptr;
	OptiXCUDABuffer<PotentialContribution> *m_ConnectData = nullptr;
	OptiXCUDABuffer<uint> *m_BlueNoise = nullptr;

	CUDABuffer<rfw::DeviceAreaLight> *m_AreaLights = nullptr;
	CUDABuffer<rfw::DevicePointLight> *m_PointLights = nullptr;
	CUDABuffer<rfw::DeviceSpotLight> *m_SpotLights = nullptr;
	CUDABuffer<rfw::DeviceDirectionalLight> *m_DirectionalLights = nullptr;

	CUDABuffer<rfw::CameraView> *m_CameraView = nullptr;
	CUDABuffer<Counters> *m_Counters = nullptr;
	std::vector<OptiXMesh *> m_Meshes;

	std::vector<optix::Transform> m_Instances;
	std::vector<rfw::TextureData> m_TexDescriptors;
	std::vector<CUDABuffer<uint> *> m_TextureBuffers;
	CUDABuffer<uint *> *m_TextureBuffersPointers = nullptr;

	OptiXCUDABuffer<glm::vec4> *m_NormalBuffer = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_AlbedoBuffer = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_InputNormalBuffer = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_InputAlbedoBuffer = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_InputPixelBuffer = nullptr;
	OptiXCUDABuffer<glm::vec4> *m_OutputPixelBuffer = nullptr;

	optix::CommandList m_DenoiseCommandList;
	optix::PostprocessingStage m_Denoiser;

	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedPrim = 0;
	float m_ProbedDistance = 0;
	glm::vec3 m_ProbedPoint = glm::vec3(0.0f);
};
