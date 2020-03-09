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
		ProbeResult(rfw::InstanceReference reference, int meshIdx, int primIdx, Triangle *t, size_t material,
					float dist)
			: object(std::move(reference)), distance(dist), materialIdx(material), meshID(meshIdx), primID(primIdx),
			  triangle(t)
		{
		}

		int meshID = 0;
		int primID = 0;
		Triangle *triangle = nullptr;
	};

	RenderSystem(const RenderSystem &other) = delete;
	RenderSystem();
	~RenderSystem();

	void load_render_api(std::string name);
	void unload_render_api();

	void set_target(GLuint *textureID, uint width, uint height);
	void set_target(rfw::utils::GLTexture *texture);
	void set_skybox(std::string filename);
	void set_skybox(rfw::utils::ArrayProxy<vec3> data, int width, int height);
	void synchronize();
	void set_animations_to(float timeInSeconds);

	GeometryReference get_geometry_ref(size_t index);
	InstanceReference get_instance_ref(size_t index);

	GeometryReference add_object(std::string fileName, int material = -1);
	GeometryReference add_object(std::string fileName, bool normalize, int material = -1);
	GeometryReference add_object(std::string fileName, bool normalize, const glm::mat4 &preTransform,
								 int material = -1);
	GeometryReference add_quad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, uint material);
	InstanceReference add_instance(const rfw::GeometryReference &geometry, glm::vec3 scaling = glm::vec3(1.0f),
								   glm::vec3 translation = glm::vec3(0.0f), float degrees = 1.0f,
								   glm::vec3 axes = glm::vec3(1.0f));
	void update_instance(const rfw::InstanceReference &instanceRef, const mat4 &transform);
	void set_animation_to(const rfw::GeometryReference &instanceRef, float timeInSeconds);

	rfw::HostMaterial get_material(size_t index) const;
	void set_material(size_t index, const rfw::HostMaterial &mat);
	int add_material(const glm::vec3 &color, float roughness = 1.0f);
	void render_frame(const Camera &camera, RenderStatus status = Converge, bool toneMap = true);

	LightReference add_point_light(const glm::vec3 &position, const glm::vec3 &radiance);
	LightReference add_spot_light(const glm::vec3 &position, float inner_deg, const glm::vec3 &radiance,
								  float outer_deg, const glm::vec3 &direction);
	LightReference add_directional_light(const glm::vec3 &direction, const glm::vec3 &radiance);

	LightReference get_area_light_ref(size_t index);
	LightReference get_point_light_ref(size_t index);
	LightReference get_spot_light_ref(size_t index);
	LightReference get_directional_light_ref(size_t index);

	void set_light_position(const LightReference &reference, const glm::vec3 &position);
	void set_light_radiance(const LightReference &reference, const glm::vec3 &radiance);
	AvailableRenderSettings get_available_settings() const;
	void set_setting(const rfw::RenderSetting &setting) const;

	void set_probe_index(glm::uvec2 pixelIdx);
	glm::uvec2 get_probe_index() const;
	ProbeResult get_probe_result();

	utils::ArrayProxy<rfw::InstanceReference> get_instances() const;
	rfw::InstanceReference *get_mutable_instances(size_t *size = nullptr);
	size_t get_instance_count() const;

	std::vector<rfw::GeometryReference> get_geometry();
	size_t get_geometry_count() const;
	rfw::RenderStats get_statistics() const;

	bool has_context() const { return m_Context != nullptr; }
	bool has_renderer() const { return m_Context != nullptr; }

  protected:
	size_t request_mesh_index();
	size_t request_instance_index();

  private:
	void update_area_lights();

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