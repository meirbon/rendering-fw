#pragma once

#include "PCH.h"

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

	std::vector<bvh::rfwMesh *> m_Meshes;
	CUDABuffer<InstanceBVHDescriptor> *m_InstanceDescriptors = nullptr;

	std::vector<CUDABuffer<glm::vec4> *> m_MeshVertices;
	std::vector<CUDABuffer<glm::uvec3> *> m_MeshIndices;
	std::vector<CUDABuffer<rfw::DeviceTriangle> *> m_MeshTriangles;
	std::vector<CUDABuffer<bvh::BVHNode> *> m_MeshBVHs;
	std::vector<CUDABuffer<bvh::MBVHNode> *> m_MeshMBVHs;
	std::vector<CUDABuffer<unsigned int> *> m_MeshBVHPrimIndices;

	// CUDABuffer<glm::vec4 *> *m_InstanceVertexPointers = nullptr;
	// CUDABuffer<glm::uvec3 *> *m_InstanceIndexPointers = nullptr;
	// CUDABuffer<bvh::BVHNode *> *m_InstanceBVHPointers = nullptr;
	// CUDABuffer<bvh::MBVHNode *> *m_InstanceMBVHPointers = nullptr;
	// CUDABuffer<uint *> *m_InstancePrimIdPointers = nullptr;

	CUDABuffer<bvh::BVHNode> *m_TopLevelCUDABVH = nullptr;
	CUDABuffer<bvh::MBVHNode> *m_TopLevelCUDAMBVH = nullptr;
	CUDABuffer<unsigned int> *m_TopLevelCUDAPrimIndices = nullptr;
	CUDABuffer<bvh::AABB> *m_TopLevelCUDAABBs = nullptr;
	CUDABuffer<glm::mat4> *m_CUDAInstanceTransforms = nullptr;
	CUDABuffer<glm::mat4> *m_CUDAInverseTransforms = nullptr;

	GLuint m_TargetID;
	GLuint m_Width, m_Height;
	bool m_InitializedGlew = false;
	TextureInterop m_CUDASurface;
	CUDABuffer<Counters> *m_Counters = nullptr;
	CUDABuffer<glm::vec4> *m_FloatTextures = nullptr;
	CUDABuffer<uint> *m_UintTextures = nullptr;
	CUDABuffer<glm::vec3> *m_Skybox = nullptr;

	std::vector<rfw::TextureData> m_TexDescriptors;
	CUDABuffer<uint *> *m_TextureBuffersPointers = nullptr;

	CUDABuffer<rfw::DeviceMaterial> *m_Materials = nullptr;
	CUDABuffer<rfw::CameraView> *m_CameraView = nullptr;
	CUDABuffer<glm::vec4> *m_Accumulator = nullptr;
	CUDABuffer<glm::vec4> *m_PathStates = nullptr;
	CUDABuffer<glm::vec4> *m_PathOrigins = nullptr;
	CUDABuffer<glm::vec4> *m_PathDirections = nullptr;
	CUDABuffer<glm::vec4> *m_PathThroughputs = nullptr;
	CUDABuffer<PotentialContribution> *m_ConnectData = nullptr;
	CUDABuffer<unsigned int> *m_BlueNoise = nullptr;

	CUDABuffer<rfw::DeviceAreaLight> *m_AreaLights = nullptr;
	CUDABuffer<rfw::DevicePointLight> *m_PointLights = nullptr;
	CUDABuffer<rfw::DeviceSpotLight> *m_SpotLights = nullptr;
	CUDABuffer<rfw::DeviceDirectionalLight> *m_DirectionalLights = nullptr;

	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedPrim = 0;
	float m_ProbedDistance = 0;
	glm::uvec2 m_ProbePixel = glm::vec2(0);
	glm::vec3 m_ProbedPoint = glm::vec3(0.0f);
	rfw::RenderStats m_Stats = {};
};

} // namespace rfw
