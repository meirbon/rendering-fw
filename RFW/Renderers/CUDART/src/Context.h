#pragma once

#include "PCH.h"

#include <memory>

namespace rfw
{

class CUDAContext : public RenderContext
{
	~CUDAContext() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> get_supported_targets() const override;

	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void render_frame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void set_materials(const std::vector<rfw::DeviceMaterial> &materials,
					   const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void set_textures(const std::vector<rfw::TextureData> &textures) override;
	void set_mesh(size_t index, const rfw::Mesh &mesh) override;
	void set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
					const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
					const rfw::DeviceDirectionalLight *directionalLights) override;
	void get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	rfw::AvailableRenderSettings get_settings() const override;
	void set_setting(const rfw::RenderSetting &setting) override;
	void update() override;
	void set_probe_index(glm::uvec2 probePos) override;
	rfw::RenderStats get_stats() const override;

  private:
	void resizeBuffers();
	void setupTexture();

	bool m_ResetFrame = true;
	bool m_Initialized = false;

	uint m_SampleIndex;
	bvh::TopLevelBVH m_TopLevelBVH = {};
	std::vector<uint> m_InstanceMeshIDs;

	std::vector<std::unique_ptr<bvh::rfwMesh>> m_Meshes;
	std::unique_ptr<CUDABuffer<InstanceBVHDescriptor>> m_InstanceDescriptors = nullptr;

	std::vector<std::unique_ptr<CUDABuffer<glm::vec4>>> m_MeshVertices;
	std::vector<std::unique_ptr<CUDABuffer<glm::uvec3>>> m_MeshIndices;
	std::vector<std::unique_ptr<CUDABuffer<rfw::DeviceTriangle>>> m_MeshTriangles;
	std::vector<std::unique_ptr<CUDABuffer<bvh::BVHNode>>> m_MeshBVHs;
	std::vector<std::unique_ptr<CUDABuffer<bvh::MBVHNode>>> m_MeshMBVHs;
	std::vector<std::unique_ptr<CUDABuffer<unsigned int>>> m_MeshBVHPrimIndices;

	// CUDABuffer<glm::vec4 *> *m_InstanceVertexPointers = nullptr;
	// CUDABuffer<glm::uvec3 *> *m_InstanceIndexPointers = nullptr;
	// CUDABuffer<bvh::BVHNode *> *m_InstanceBVHPointers = nullptr;
	// CUDABuffer<bvh::MBVHNode *> *m_InstanceMBVHPointers = nullptr;
	// CUDABuffer<uint *> *m_InstancePrimIdPointers = nullptr;

	std::unique_ptr<CUDABuffer<bvh::BVHNode>> m_TopLevelCUDABVH = nullptr;
	std::unique_ptr<CUDABuffer<bvh::MBVHNode>> m_TopLevelCUDAMBVH = nullptr;
	std::unique_ptr<CUDABuffer<unsigned int>> m_TopLevelCUDAPrimIndices = nullptr;
	std::unique_ptr<CUDABuffer<glm::mat4>> m_CUDAInstanceTransforms = nullptr;
	std::unique_ptr<CUDABuffer<glm::mat4>> m_CUDAInverseTransforms = nullptr;

	GLuint m_TargetID;
	GLuint m_Width, m_Height;
	bool m_InitializedGlew = false;
	TextureInterop m_CUDASurface;
	std::unique_ptr<CUDABuffer<Counters>> m_Counters = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_FloatTextures = nullptr;
	std::unique_ptr<CUDABuffer<uint>> m_UintTextures = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec3>> m_Skybox = nullptr;

	std::vector<rfw::TextureData> m_TexDescriptors;
	std::unique_ptr<CUDABuffer<uint *>> m_TextureBuffersPointers = nullptr;

	std::unique_ptr<CUDABuffer<rfw::DeviceMaterial>> m_Materials = nullptr;
	std::unique_ptr<CUDABuffer<rfw::CameraView>> m_CameraView = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_Accumulator = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_PathStates = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_PathOrigins = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_PathDirections = nullptr;
	std::unique_ptr<CUDABuffer<glm::vec4>> m_PathThroughputs = nullptr;
	std::unique_ptr<CUDABuffer<PotentialContribution>> m_ConnectData = nullptr;
	std::unique_ptr<CUDABuffer<unsigned int>> m_BlueNoise = nullptr;

	std::unique_ptr<CUDABuffer<rfw::DeviceAreaLight>> m_AreaLights = nullptr;
	std::unique_ptr<CUDABuffer<rfw::DevicePointLight>> m_PointLights = nullptr;
	std::unique_ptr<CUDABuffer<rfw::DeviceSpotLight>> m_SpotLights = nullptr;
	std::unique_ptr<CUDABuffer<rfw::DeviceDirectionalLight>> m_DirectionalLights = nullptr;

	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedPrim = 0;
	float m_ProbedDistance = 0;
	glm::uvec2 m_ProbePixel = glm::vec2(0);
	glm::vec3 m_ProbedPoint = glm::vec3(0.0f);
	rfw::RenderStats m_Stats = {};
};

} // namespace rfw
