#pragma once

#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <bitset>
#include <mutex>
#include <future>

#include <Structures.h>
#include <RenderContext.h>
#include <ContextExport.h>

#include "Camera.h"
#include "Settings.h"
#include "Skybox.h"
#include "utils/Window.h"
#include "utils/String.h"

#include "MaterialList.h"
#include "SceneTriangles.h"
#include "utils/gl/GLShader.h"
#include "utils/ThreadPool.h"
#include "utils/Averager.h"

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
		std::vector<char> buffer(1024, 0);
		va_list arg;
		va_start(arg, format);
		utils::string::format(buffer, format, arg);
		va_end(arg);
		m_Message = std::string(buffer.data());
	}

	[[nodiscard]] const char *what() const noexcept override { return m_Message.c_str(); }

  private:
	std::string m_Message;
};

// Use lightweight object as a geometry reference for now, we might want to expand in the future
class GeometryReference
{

	friend class rfw::RenderSystem;

  private:
	GeometryReference(size_t index, rfw::RenderSystem &system) : m_Index(index), m_System(&system) { assert(m_System); }

  public:
	GeometryReference() = default;
	explicit operator size_t() const { return static_cast<size_t>(m_Index); }
	[[nodiscard]] size_t getIndex() const { return m_Index; }

	[[nodiscard]] bool isAnimated() const;
	void setAnimationTime(float time) const;
	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const;
	[[nodiscard]] const std::vector<glm::mat4> &getMeshMatrices() const;
	[[nodiscard]] const std::vector<std::vector<int>> &getLightIndices() const;

  protected:
	SceneTriangles *getObject() const;

  private:
	size_t m_Index; // Loaded geometry index
	rfw::RenderSystem *m_System;
};

class LightReference
{
	friend class rfw::RenderSystem;

  public:
	enum LightType
	{
		AREA = 0,
		POINT = 1,
		SPOT = 2,
		DIRECTIONAL = 3
	};

	LightReference(size_t index, LightType type, rfw::RenderSystem &system) : m_Index(index), m_System(&system), m_Type(type) { assert(m_System); }
	LightReference() = default;

	explicit operator size_t() const { return m_Index; }
	[[nodiscard]] size_t getIndex() const { return m_Index; }
	[[nodiscard]] LightType getType() const { return m_Type; }

  private:
	size_t m_Index;
	rfw::RenderSystem *m_System;
	LightType m_Type;
};

// Use lightweight object as an instance reference for now, we might want to expand in the future
class InstanceReference
{
	friend class rfw::RenderSystem;

  public:
	InstanceReference() = default;
	InstanceReference(size_t index, GeometryReference reference, rfw::RenderSystem &system);

	explicit operator size_t() const { return m_Members->index; }

	[[nodiscard]] const GeometryReference &getGeometryReference() const { return m_Members->geomReference; }

	void setTranslation(glm::vec3 value);
	void setRotation(float degrees, glm::vec3 axis);
	void setRotation(const glm::quat &q);
	void setRotation(const glm::vec3 &euler);
	void setScaling(glm::vec3 value);

	void translate(glm::vec3 offset);
	void rotate(float degrees, glm::vec3 axis);
	void scale(glm::vec3 offset);
	void update() const;

	[[nodiscard]] size_t getIndex() const { return m_Members->index; }
	[[nodiscard]] const std::vector<size_t> &getIndices() const { return m_Members->instanceIDs; }

	[[nodiscard]] glm::mat4 getMatrix() const;
	[[nodiscard]] glm::mat3 getInverseMatrix() const;

	[[nodiscard]] glm::vec3 getScaling() const { return m_Members->scaling; }
	[[nodiscard]] glm::vec3 getRotation() const { return glm::eulerAngles(m_Members->rotation); }
	[[nodiscard]] glm::vec3 getTranslation() const { return m_Members->translation; }

  private:
	struct Members
	{
		explicit Members(const GeometryReference &ref);
		glm::vec3 translation = glm::vec3(0);
		glm::quat rotation = glm::identity<glm::quat>();
		glm::vec3 scaling = glm::vec3(1);
		size_t index;
		std::vector<size_t> instanceIDs;
		GeometryReference geomReference;
		rfw::RenderSystem *rSystem = nullptr;
	};
	std::shared_ptr<Members> m_Members;
};

struct AABB
{
	AABB()
	{
		mMin = vec3(1e34f);
		mMax = vec3(-1e34f);
	}
	glm::vec3 mMin{}, mMax{};

	void grow(vec3 p)
	{
		mMin = min(mMin, p);
		mMax = max(mMax, p);
	}
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
		rfw::InstanceReference object;
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

	void loadRenderAPI(const std::string_view &name);
	void unloadRenderAPI();

	void setTarget(GLuint *textureID, uint width, uint height);
	void setTarget(rfw::utils::GLTexture *texture);
	void setSkybox(std::string_view filename);
	void synchronize();
	void updateAnimationsTo(float timeInSeconds);

	rfw::GeometryReference getGeometryReference(size_t index);
	rfw::InstanceReference getInstanceReference(size_t index);

	rfw::GeometryReference addObject(std::string_view fileName, int material = -1);
	rfw::GeometryReference addObject(std::string_view fileName, bool normalize, int material = -1);
	rfw::GeometryReference addObject(std::string_view fileName, bool normalize, const glm::mat4 &preTransform, int material = -1);
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
	LightReference addSpotLight(const glm::vec3 &position, float cosInner, const glm::vec3 &radiance, float cosOuter, const glm::vec3 &direction);
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

	[[nodiscard]] AABB calculateSceneBounds() const;

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

	rfw::Skybox *m_Skybox = nullptr;
	rfw::MaterialList *m_Materials = nullptr;
	rfw::RenderContext *m_Context = nullptr;

	std::bitset<32> m_Changed;
	std::vector<bool> m_ModelChanged;
	std::vector<bool> m_InstanceChanged;
	std::vector<SceneTriangles *> m_Models;
	std::mutex m_SetMeshMutex;

	// Vector containing the index of (instance, object, mesh) for probe retrieval
	std::vector<std::tuple<int, int, int>> m_InverseInstanceMapping;

	std::vector<InstanceReference> m_Instances;
	std::vector<glm::mat4> m_InstanceMatrices;

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