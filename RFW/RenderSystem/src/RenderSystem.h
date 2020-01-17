#pragma once

#include "rfw.h"

namespace rfw
{
namespace utils
{
class GLTexture;
}

class MaterialList;
class HostMaterial;
class AssimpObject;
class RenderSystem;

class RfwException : public std::exception
{
  public:
	explicit RfwException(const char *format, ...)
	{
		std::vector<char> buffer(2048, 0);
		va_list arg;
		va_start(arg, format);
		utils::string::format(buffer.data(), format, arg);
		va_end(arg);
		m_Message = std::string(buffer.data());
	}

	[[nodiscard]] const char *what() const noexcept override { return m_Message.c_str(); }

  private:
	std::string m_Message;
};

class RenderSystem
{
	friend class GeometryReference;
	friend class InstanceReference;
	friend class SceneTriangles;
	friend class AssimpObject;
	friend class gLTFObject;
	friend class Quad;

  public:
	class ProbeResult
	{
		friend class RenderSystem;

	  public:
		ProbeResult() = default;
		InstanceReference object;
		float distance = 0;
		size_t materialIdx = 0;

	  private:
		ProbeResult(rfw::InstanceReference reference, int meshIdx, int primIdx, Triangle *t, size_t material, float dist)
			: object(std::move(reference)), distance(dist), materialIdx(material), meshID(meshIdx), primID(primIdx), triangle(t)
		{
		}

		int meshID = 0;
		int primID = 0;
		Triangle *triangle = nullptr;
	};

	RenderSystem();
	~RenderSystem();

	void loadRenderAPI(std::string name);
	void unloadRenderAPI();

	void setTarget(GLuint *textureID, uint width, uint height);
	void setTarget(rfw::utils::GLTexture *texture);
	void setSkybox(std::string filename);
	void synchronize();
	void updateAnimationsTo(float timeInSeconds);

	rfw::GeometryReference getGeometryReference(size_t index);
	rfw::InstanceReference getInstanceReference(size_t index);

	rfw::GeometryReference addObject(std::string fileName, int material = -1);
	rfw::GeometryReference addObject(std::string fileName, bool normalize, int material = -1);
	rfw::GeometryReference addObject(std::string fileName, bool normalize, const glm::mat4 &preTransform, int material = -1);
	rfw::GeometryReference addQuad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, uint material);
	rfw::InstanceReference addInstance(const rfw::GeometryReference &geometry, glm::vec3 scaling = glm::vec3(1.0f), glm::vec3 translation = glm::vec3(0.0f),
									   float degrees = 1.0f, glm::vec3 axes = glm::vec3(1.0f));
	void updateInstance(const rfw::InstanceReference &instanceRef, const mat4 &transform);
	void setAnimationTime(const rfw::GeometryReference &instanceRef, float timeInSeconds);

	rfw::HostMaterial getMaterial(size_t index) const;
	void setMaterial(size_t index, const rfw::HostMaterial &mat);
	int addMaterial(const glm::vec3 &color, float roughness = 1.0f);
	void renderFrame(const Camera &camera, RenderStatus status = Converge, bool toneMap = true);

	LightReference addPointLight(const glm::vec3 &position, const glm::vec3 &radiance);
	LightReference addSpotLight(const glm::vec3 &position, float inner_deg, const glm::vec3 &radiance, float outer_deg, const glm::vec3 &direction);
	LightReference addDirectionalLight(const glm::vec3 &direction, const glm::vec3 &radiance);

	LightReference getAreaLightReference(size_t index);
	LightReference getPointLightReference(size_t index);
	LightReference getSpotLightReference(size_t index);
	LightReference getDirectionalLightReference(size_t index);

	void setPosition(const LightReference &reference, const glm::vec3 &position);
	void setRadiance(const LightReference &reference, const glm::vec3 &radiance);
	void setEnergy(const LightReference &reference, float energy);
	rfw::AvailableRenderSettings getAvailableSettings() const;
	void setSetting(const rfw::RenderSetting &setting) const;

	void setProbeIndex(glm::uvec2 pixelIdx);
	glm::uvec2 getProbeIndex() const;
	ProbeResult getProbeResult();

	const std::vector<rfw::InstanceReference> &getInstances() const;
	rfw::InstanceReference *getMutableInstances();
	size_t getInstanceCount() const;

	std::vector<rfw::GeometryReference> getGeometry();
	size_t getGeometryCount() const;
	rfw::RenderStats getRenderStats() const;
	bool hasContext() const { return m_Context != nullptr; }
	bool hasRenderer() const { return m_Context != nullptr; }

  protected:
	size_t requestMeshIndex();
	size_t requestInstanceIndex();

  private:
	void updateAreaLights();

	utils::ThreadPool m_ThreadPool;
	GLuint m_TargetID = 0, m_FrameBufferID = 0;
	GLuint m_TargetWidth = 0, m_TargetHeight = 0;
	size_t m_EmptyMeshSlots = 0;
	size_t m_EmptyInstanceSlots = 0;
	std::future<void> m_UpdateThread, m_AnimationsThread;

	std::vector<bool> m_MeshSlots;
	std::vector<bool> m_InstanceSlots;

	bool m_ShouldReset = true, m_UninitializedMeshes = true;
	enum Changed
	{
		MODELS = 0,
		INSTANCES = 1,
		MATERIALS = 2,
		SKYBOX = 3,
		LIGHTS = 4,
		AREA_LIGHTS = 5,
		ANIMATED = 6
	};

	glm::uvec2 m_ProbeIndex = glm::uvec2(0, 0);
	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedPrimitive = 0;
	float m_ProbeDistance = 0.0f;
	utils::Averager<float, 32> m_AnimationStat;
	utils::Averager<float, 32> m_RenderStat;

	Skybox m_Skybox;
	MaterialList *m_Materials = nullptr;
	RenderContext *m_Context = nullptr;

	std::bitset<32> m_Changed;
	std::vector<bool> m_ModelChanged;
	std::vector<bool> m_InstanceChanged;
	std::vector<SceneTriangles *> m_Models;

	// Vector containing the index of (instance, object, mesh) for probe retrieval
	std::vector<std::tuple<int, int, int>> m_InverseInstanceMapping;

	std::vector<InstanceReference> m_Instances;
	std::vector<simd::matrix4> m_InstanceMatrices;

	std::vector<std::vector<std::vector<int>>> m_ObjectLightIndices;
	std::vector<float> m_AreaLightEnergy;
	std::vector<std::pair<uint, uint>> m_ObjectMaterialRange;

	std::vector<AreaLight> m_AreaLights;
	std::vector<PointLight> m_PointLights;
	std::vector<SpotLight> m_SpotLights;
	std::vector<DirectionalLight> m_DirectionalLights;

	utils::GLShader m_ToneMapShader;

	CreateContextFunction m_CreateContextFunction = nullptr;
	DestroyContextFunction m_DestroyContextFunction = nullptr;
	void *m_ContextModule = nullptr;
};

} // namespace rfw