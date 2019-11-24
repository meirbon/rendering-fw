#pragma once

#include <vector>
#include <string>
#include <string_view>
#include <bitset>
#include <mutex>

#include <Structures.h>
#include <RenderContext.h>
#include <ContextExport.h>

#include "Camera.h"
#include "Settings.h"
#include "Skybox.h"
#include "utils/Window.h"

#include "MaterialList.h"
#include "SceneTriangles.h"

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

// Use lightweight object as a geometry reference for now, we might want to expand in the future
class GeometryReference
{
	friend class rfw::RenderSystem;

  private:
	GeometryReference(size_t index, rfw::RenderSystem &system) : m_Index(index), m_System(&system) { assert(m_System); }

  public:
	operator uint() const { return static_cast<uint>(m_Index); }
	operator size_t() const { return m_Index; }
	[[nodiscard]] size_t getIndex() const { return m_Index; }

	void setAnimationTime(float time) const;
	rfw::Mesh getMesh() const;

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

	operator uint() const { return static_cast<uint>(m_Index); }
	operator size_t() const { return m_Index; }
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
	InstanceReference(size_t index, GeometryReference reference, rfw::RenderSystem &system)
	{
		m_Members = std::make_shared<Members>(reference);
		m_Members->m_Index = index;
		m_Members->m_Reference = reference;
		m_Members->m_System = &system;
		assert(m_Members->m_System);
		m_Members->m_Translation = glm::vec3(0.0f);
		m_Members->m_Rotation = glm::identity<glm::quat>();
		m_Members->m_Scaling = glm::vec3(1.0f);
	}

	operator uint() const { return static_cast<uint>(m_Members->m_Index); }
	operator size_t() const { return m_Members->m_Index; }

	[[nodiscard]] const GeometryReference &getReference() const { return m_Members->m_Reference; }

	void setTranslation(glm::vec3 value);
	void setRotation(float degrees, glm::vec3 axis);
	void setRotation(const glm::quat &q);
	void setScaling(glm::vec3 value);

	void translate(glm::vec3 offset);
	void rotate(float degrees, glm::vec3 axis);
	void scale(glm::vec3 offset);
	void update() const;
	[[nodiscard]] size_t getIndex() const { return m_Members->m_Index; }

	[[nodiscard]] glm::mat4 getMatrix() const;
	[[nodiscard]] glm::mat3 getInverseMatrix() const;

	[[nodiscard]] glm::vec3 getScaling() const { return m_Members->m_Scaling; }
	[[nodiscard]] glm::quat getRotation() const { return m_Members->m_Rotation; }
	[[nodiscard]] glm::vec3 getTranslation() const { return m_Members->m_Translation; }

  private:
	struct Members
	{
		explicit Members(const GeometryReference &ref);
		glm::vec3 m_Translation;
		glm::quat m_Rotation;
		glm::vec3 m_Scaling;
		size_t m_Index;
		GeometryReference m_Reference;
		rfw::RenderSystem *m_System;
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

  public:
	class ProbeResult
	{
		friend class RenderSystem;

	  public:
		rfw::InstanceReference object;
		float distance;
		size_t materialIdx;

	  private:
		ProbeResult() = default;
		ProbeResult(const rfw::InstanceReference &reference, size_t primIdx, size_t material, float dist)
			: object(reference), distance(dist), primitiveIndex(primIdx), materialIdx(material)
		{
		}
		size_t primitiveIndex;
	};

	RenderSystem();
	~RenderSystem();

	void loadRenderAPI(const std::string_view &name);
	void unloadRenderAPI();

	void setTarget(std::shared_ptr<rfw::utils::Window> m_Window);
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
	rfw::GeometryReference addQuad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, const uint material);
	rfw::InstanceReference addInstance(const rfw::GeometryReference &geometry, glm::vec3 scaling = glm::vec3(1.0f), glm::vec3 translation = glm::vec3(0.0f),
									   float degrees = 1.0f, glm::vec3 axes = glm::vec3(1.0f));
	void updateInstance(const rfw::InstanceReference &instanceRef, const mat4 &transform);
	void setAnimationTime(const rfw::GeometryReference &instanceRef, float timeInSeconds);

	rfw::HostMaterial getMaterial(size_t index);
	void setMaterial(size_t index, const rfw::HostMaterial &mat);
	uint addMaterial(const glm::vec3 &color, float roughness = 1.0f);
	void renderFrame(const Camera &camera, RenderStatus status = Converge);

	LightReference addPointLight(const glm::vec3 &position, float energy, const glm::vec3 &radiance);
	LightReference addSpotLight(const glm::vec3 &position, float cosInner, const glm::vec3 &radiance, float cosOuter, float energy, const glm::vec3 &direction);
	LightReference addDirectionalLight(const glm::vec3 &direction, float energy, const glm::vec3 &radiance);

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

  private:
	void updateAreaLights();

	bool m_ShouldReset = true, m_UninitializedMeshes = true;
	enum Changed
	{
		MODELS = 0,
		INSTANCES = 1,
		MATERIALS = 2,
		SKYBOX = 3,
		LIGHTS = 4,
		AREA_LIGHTS = 5
	};

	glm::uvec2 m_ProbeIndex = glm::uvec2(0, 0);
	unsigned int m_ProbedInstance = 0;
	unsigned int m_ProbedPrimitive = 0;
	float m_ProbeDistance = 0.0f;

	rfw::Skybox *m_Skybox = nullptr;
	rfw::MaterialList *m_Materials = nullptr;
	rfw::RenderContext *m_Context = nullptr;

	std::bitset<32> m_Changed;
	std::vector<bool> m_ModelChanged;
	std::vector<bool> m_InstanceChanged;
	std::vector<SceneTriangles *> m_Models;
	std::mutex m_SetMeshMutex;

	std::vector<InstanceReference> m_Instances;
	std::vector<glm::mat4> m_InstanceMatrices;

	std::vector<std::vector<uint>> m_ObjectLightIndices;
	std::vector<float> m_AreaLightEnergy;
	std::vector<std::pair<uint, uint>> m_ObjectMaterialRange;

	std::vector<AreaLight> m_AreaLights;
	std::vector<PointLight> m_PointLights;
	std::vector<SpotLight> m_SpotLights;
	std::vector<DirectionalLight> m_DirectionalLights;

	CreateContextFunction m_CreateContextFunction = nullptr;
	DestroyContextFunction m_DestroyContextFunction = nullptr;
	void *m_ContextModule = nullptr;
};

} // namespace rfw