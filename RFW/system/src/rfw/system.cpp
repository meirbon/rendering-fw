#include "rfw.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <libloaderapi.h>
#include <rfw/utils/timer.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#else
static_assert(false, "Platform not supported");
#endif

#ifdef WIN32
extern "C"
{
	__declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
}

extern "C"
{
	__declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}
#endif

#define USE_TINY_GLTF 1
#define ENABLE_THREADING 1

using namespace rfw;
using namespace utils;

rfw::system::system()
	: m_ThreadPool(std::thread::hardware_concurrency()),
	  m_ToneMapShader("shaders/draw-tex.vert", "shaders/tone-map.frag")
{
	m_Materials = new material_list();
	glGenFramebuffers(1, &m_FrameBufferID);
	m_ToneMapShader.bind();
	m_ToneMapShader.set_uniform("view", mat4(1.0f));
	m_ToneMapShader.unbind();

	m_TargetWidth = 0;
	m_TargetHeight = 0;
}

rfw::system::~system()
{
	m_ThreadPool.stop(true);

	if (m_Context)
		unload_render_api();

	for (auto *object : m_Models)
		delete object;

	delete m_Materials;
	m_Materials = nullptr;

	glDeleteFramebuffers(1, &m_FrameBufferID);
	m_FrameBufferID = 0;
}

void *LoadModule(const char *file)
{
	void *module = nullptr;
#ifdef WIN32
	module = LoadLibrary(file);
	if (!module)
		fprintf(stderr, "Loading library \"%s\" error: %lu\n", file, GetLastError());
#else
	module = dlopen(file, RTLD_NOW);
	if (!module)
		fprintf(stderr, "%s\n", dlerror());
#endif

	return module;
}

void printLoadError()
{
#ifdef WIN32
	fprintf(stderr, "%lu\n", GetLastError());
#else
	fprintf(stderr, "%s\n", dlerror());
#endif
}

void *LoadModuleFunction(void *module, const char *funcName)
{
#ifdef WIN32
	return static_cast<void *>(GetProcAddress(HMODULE(module), funcName));
#else
	return static_cast<void *>(dlsym(module, funcName));
#endif
}

static_assert(sizeof(Material) == sizeof(DeviceMaterial), "Material structs are not same size.");
static_assert(sizeof(Triangle) == sizeof(DeviceTriangle), "Triangle structs are not same size.");
static_assert(sizeof(AreaLight) == sizeof(DeviceAreaLight), "Area light structs are not same size.");
static_assert(sizeof(PointLight) == sizeof(DevicePointLight), "Point light structs are not same size.");
static_assert(sizeof(SpotLight) == sizeof(DeviceSpotLight), "Spot light structs are not same size.");
static_assert(sizeof(DirectionalLight) == sizeof(DeviceDirectionalLight),
			  "Directional light structs are not same size.");

void rfw::system::load_render_api(std::string name)
{
#ifdef WIN32
	const std::string_view extension = ".dll"; // Windows shared library
#elif __linux__
	const std::string_view extension = ".so"; // Linux shared library
#elif __APPLE__
	const std::string_view extension = ".dylib"; // MacOS shared library
#else
#pragma error
#endif

	const std::string cwd = utils::file::get_working_path();
	const std::string libName = name + std::string(extension);
	const std::string libPath = cwd + '/' + libName;

	if (m_Context)
	{
		throw std::runtime_error(
			"A RenderContext was already loaded, unload current context before loading a new context.");
	}

	if (!utils::file::exists(libPath))
	{
		throw std::runtime_error("Library does not exist.");
	}

	m_ContextModule = LoadModule(libPath.data());
	if (!m_ContextModule)
	{
		const std::string message = std::string("Could not load library: ") + libPath;
		throw std::runtime_error(message);
	}

	const auto createPtr = LoadModuleFunction(m_ContextModule, CREATE_RENDER_CONTEXT_FUNC_NAME);
	const auto destroyPtr = LoadModuleFunction(m_ContextModule, DESTROY_RENDER_CONTEXT_FUNC_NAME);
	m_CreateContextFunction = CreateContextFunction(createPtr);
	m_DestroyContextFunction = DestroyContextFunction(destroyPtr);

	if (!m_CreateContextFunction)
	{
		printLoadError();
		FAILURE("Could not load create context function from shared library.");
	}
	else if (!m_DestroyContextFunction)
	{
		printLoadError();
		FAILURE("Could not load destroy context function from shared library.");
	}

	m_Context = m_CreateContextFunction();
}

void rfw::system::unload_render_api()
{
	if (m_Context)
	{
		m_Context->cleanup();
		m_DestroyContextFunction(m_Context);
	}
	if (m_ContextModule)
	{
#ifdef WIN32
		FreeLibrary(HMODULE(m_ContextModule));
#else
		dlclose(m_ContextModule);
#endif
	}

	m_Context = nullptr;
	m_ContextModule = nullptr;
}

void system::set_target(GLuint *textureID, uint width, uint height)
{
	assert(textureID != nullptr && *textureID != 0);

	m_Context->init(textureID, width, height);
	m_TargetID = *textureID;
	m_TargetWidth = width;
	m_TargetHeight = height;

	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_TargetID);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CheckGL();
}

void system::set_target(rfw::utils::texture *texture)
{
	assert(texture != nullptr);

	if (texture == nullptr)
		throw std::runtime_error("Invalid texture.");
	try
	{
		m_Context->init(&texture->m_ID, texture->get_width(), texture->get_height());
	}
	catch (const std::exception &e)
	{
		FAILURE("%s", e.what());
	}

	m_TargetID = texture->m_ID;
	m_TargetWidth = texture->get_width();
	m_TargetHeight = texture->get_height();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CheckGL();
}

void system::set_skybox(std::string filename)
{

	if (m_Skybox.get_source() == std::string(filename))
		return;

	if (!file::exists(filename))
	{
		WARNING("File: %s does not exist.", filename.data());
		throw std::runtime_error("Skybox file does not exist.");
	}

	m_Skybox.load(filename);
	m_Changed[SKYBOX] = true;
}

void system::set_skybox(rfw::utils::array_proxy<vec3> data, int width, int height)
{
	m_Skybox.set(data, width, height);
	m_Changed[SKYBOX] = true;
}

void system::synchronize()
{
	if (m_AnimationsThread.valid())
		m_AnimationsThread.get();

	if (m_Changed[SKYBOX])
		m_Context->set_sky(m_Skybox.get_buffer(), m_Skybox.get_width(), m_Skybox.get_height());

	if (m_Materials->is_dirty())
		m_Materials->generate_device_materials();

	// Update materials
	if (!m_Materials->get_materials().empty() && m_Changed[MATERIALS])
	{
		const auto &textures = m_Materials->get_texture_descriptors();
		if (!textures.empty())
			m_Context->set_textures(textures);

		m_Context->set_materials(m_Materials->get_device_materials(), m_Materials->get_material_tex_ids());
	}

	// First update area lights as these might update models in the process.
	// Updating area lights first prevents us from having to update models twice
	if (m_Changed[AREA_LIGHTS] && !m_UninitializedMeshes)
	{
		update_area_lights();
		m_Changed[LIGHTS] = true;
		m_Changed[AREA_LIGHTS] = false;
	}

	if (m_Changed[ANIMATED])
	{
		for (rfw::geometry::SceneTriangles *object : m_Models)
		{
			if (!object->is_animated())
				continue;

			const auto changedMeshes = object->get_changed_meshes();
			auto meshes = object->get_meshes();

			for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
			{
				if (!changedMeshes[i])
					continue;
				const auto &[index, mesh] = meshes[i];
				m_Context->set_mesh(index, mesh);
			}
		}
	}

	// Update loaded objects/models
	if (m_Changed[MODELS])
	{
		for (size_t i = 0; i < m_Models.size(); i++)
		{
			// Only update if changed
			if (!m_ModelChanged[i])
				continue;

			for (const auto &[meshSlot, mesh] : m_Models[i]->get_meshes())
			{
				assert(mesh.vertexCount > 0);
				assert(mesh.triangleCount > 0);
				assert(mesh.vertices);
				assert(mesh.normals);
				assert(mesh.triangles);

				m_Context->set_mesh(meshSlot, mesh);
			}

			// Reset state
			m_ModelChanged[i] = false;
		}
	}

	if (m_Changed[AREA_LIGHTS])
	{
		update_area_lights();
		m_Changed[LIGHTS] = true;
	}

	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		rfw::instance_ref &instRef = m_Instances[i];

		if (m_InstanceChanged[i]) // Need to update this instance anyway
		{
			// Update instance
			const auto &matrix = m_InstanceMatrices[i];
			const auto geometry = instRef.get_geometry_ref();

			const auto &instanceMapping = instRef.getIndices();
			const auto &meshes = geometry.get_meshes();
			const auto &matrices = geometry.get_mesh_matrices();

			for (int j = 0, sj = static_cast<int>(meshes.size()); j < sj; j++)
			{
				const auto meshID = meshes[j].first;
				const auto instanceID = instanceMapping[j];
				const simd::matrix4 transform = matrix * matrices[j];
				const mat3 inverse_transform = transform.inversed().transposed().matrix;
				m_Context->set_instance(instanceID, meshID, transform.matrix, inverse_transform);
			}

			m_InstanceChanged[i] = false;
		}
		else
		{
			const auto geometry = instRef.get_geometry_ref();
			if (!geometry.is_animated())
				continue;

			const auto object = instRef.get_geometry_ref().get_object();
			const auto changedTransforms = object->get_changed_matrices();

			// Update instance
			const auto &matrix = m_InstanceMatrices[i];

			const auto &instanceMapping = instRef.getIndices();
			const auto &meshes = geometry.get_meshes();
			const auto &matrices = geometry.get_mesh_matrices();

			for (int j = 0, sj = static_cast<int>(meshes.size()); j < sj; j++)
			{
				if (!changedTransforms[j])
					continue;

				const auto meshID = meshes[j].first;
				const auto instanceID = instanceMapping[j];
				const simd::matrix4 transform = matrix * matrices[j];
				const mat3 inverse_transform = mat3(transform.inversed().transposed().matrix);
				m_Context->set_instance(instanceID, meshID, transform.matrix, inverse_transform);
			}
		}
	}

	// Update instances
	if (m_Changed[INSTANCES])
	{
		for (size_t i = 0; i < m_Instances.size(); i++)
		{
			// Only update if changed
			if (!m_InstanceChanged.at(i))
				continue;

			// Update instance
			const auto &ref = m_Instances[i];
			const auto &matrix = m_InstanceMatrices[i];

			const auto &instanceMapping = ref.getIndices();
			const auto &meshes = ref.get_geometry_ref().get_meshes();
			const auto &matrices = ref.get_geometry_ref().get_mesh_matrices();

			for (int j = 0, sj = static_cast<int>(meshes.size()); j < sj; j++)
			{
				const auto meshID = meshes[j].first;
				const auto instanceID = instanceMapping[j];
				const simd::matrix4 transform = matrix * matrices[j];
				const mat3 inverse_transform = mat3(transform.inversed().transposed().matrix);
				m_Context->set_instance(instanceID, meshID, transform.matrix, inverse_transform);
			}
		}
	}

	if (m_Changed[LIGHTS])
	{
		LightCount count = {};
		count.areaLightCount = static_cast<uint>(m_AreaLights.size());
		count.pointLightCount = static_cast<uint>(m_PointLights.size());
		count.spotLightCount = static_cast<uint>(m_SpotLights.size());
		count.directionalLightCount = static_cast<uint>(m_DirectionalLights.size());
		m_Context->set_lights(count, reinterpret_cast<const DeviceAreaLight *>(m_AreaLights.data()),
							  reinterpret_cast<const DevicePointLight *>(m_PointLights.data()),
							  reinterpret_cast<const DeviceSpotLight *>(m_SpotLights.data()),
							  reinterpret_cast<const DeviceDirectionalLight *>(m_DirectionalLights.data()));
	}

	if (m_Changed.any())
	{
		m_Context->update();
		m_ShouldReset = true;
		m_Changed.reset();

		for (auto &&i : m_InstanceChanged)
			i = false;
	}
}

void system::set_animations_to(const float timeInSeconds)
{
	timer t = {};
#if ENABLE_THREADING
	std::vector<std::future<void>> updates;
	updates.reserve(m_Models.size());

	for (rfw::geometry::SceneTriangles *object : m_Models)
	{
		if (object->is_animated())
			updates.push_back(m_ThreadPool.push([object, timeInSeconds](int) { object->set_time(timeInSeconds); }));
	}

	if (!updates.empty())
	{
		m_Changed[ANIMATED] = true;
		m_ShouldReset = true;

		for (std::future<void> &update : updates)
		{
			if (update.valid())
				update.get();
		}
	}
#else
	for (auto object : m_Models)
	{
		if (object->is_animated())
		{
			m_Changed[ANIMATED] = true;
			m_ShouldReset = true;
			object->set_time(timeInSeconds);
		}
	}
#endif
	m_AnimationStat.add_sample(t.elapsed());
}

geometry_ref system::get_geometry_ref(size_t index)
{
	if (index >= m_Models.size())
		throw std::runtime_error("Geometry at given index does not exist.");

	return rfw::geometry_ref(index, *this);
}

instance_ref system::get_instance_ref(size_t index)
{
	if (index >= m_Instances.size())
		throw std::runtime_error("Instance at given index does not exist.");

	return m_Instances[index];
}

geometry_ref system::add_object(std::string fileName, int material)
{
	return add_object(std::move(fileName), false, glm::mat4(1.0f), material);
}

geometry_ref system::add_object(std::string fileName, bool normalize, int material)
{
	return add_object(std::move(fileName), normalize, glm::mat4(1.0f), material);
}

geometry_ref system::add_object(std::string fileName, bool normalize, const glm::mat4 &preTransform, int material)
{
	if (!utils::file::exists(fileName))
		throw LoadException(fileName);

	const size_t idx = m_Models.size();
	const size_t matFirst = m_Materials->size();

	// Add model to list
	rfw::geometry::SceneTriangles *triangles = nullptr;

#if USE_TINY_GLTF
	if (utils::string::ends_with(fileName.data(), {std::string(".gltf"), std::string(".glb")}))
		triangles =
			new rfw::geometry::gltf::Object(fileName, m_Materials, static_cast<uint>(idx), preTransform, material);
	else
#endif
		triangles = new rfw::geometry::assimp::Object(fileName, m_Materials, static_cast<uint>(idx), preTransform,
													  normalize, material);

	triangles->prepare_meshes(*this);
	assert(!triangles->get_meshes().empty());

	m_Models.push_back(triangles);
	m_ModelChanged.push_back(true);

	const auto lightFlags = m_Materials->get_material_light_flags();

	const auto lightIndices = m_Models[idx]->get_light_indices(lightFlags, true);
	assert(lightIndices.size() == m_Models[idx]->get_meshes().size());

	for (const auto &mlIndices : lightIndices)
	{
		if (!mlIndices.empty())
		{
			m_Changed[AREA_LIGHTS] = true;
			m_Changed[LIGHTS] = true;
		}
	}

	m_ObjectLightIndices.push_back(lightIndices);
	if (material != 0)
		m_ObjectMaterialRange.emplace_back(static_cast<uint>(matFirst),
										   static_cast<uint>(m_Materials->get_materials().size()));
	else
		m_ObjectMaterialRange.emplace_back(static_cast<uint>(material), static_cast<uint>(material + 1));

	// Update flags
	m_Changed[MODELS] = true;
	m_Changed[MATERIALS] = true;

	m_UninitializedMeshes = true;

	// Return reference
	return geometry_ref(idx, *this);
}

rfw::geometry_ref system::add_quad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height,
								   const uint material)
{
	const size_t idx = m_Models.size();
	if (m_Materials->get_materials().size() <= material)
		throw LoadException("Material does not exist.");

	// Add model to list
	rfw::geometry::SceneTriangles *triangles = new rfw::geometry::Quad(N, pos, width, height, material);

	triangles->prepare_meshes(*this);
	assert(!triangles->get_meshes().empty());

	m_Models.push_back(triangles);
	m_ModelChanged.push_back(true);

	const auto lightFlags = m_Materials->get_material_light_flags();

	const auto lightIndices = m_Models[idx]->get_light_indices(lightFlags, true);
	assert(lightIndices.size() == m_Models[idx]->get_meshes().size());

	for (const auto &mlIndices : lightIndices)
	{
		if (!mlIndices.empty())
		{
			m_Changed[AREA_LIGHTS] = true;
			m_Changed[LIGHTS] = true;
		}
	}

	m_ObjectLightIndices.push_back(lightIndices);
	m_ObjectMaterialRange.emplace_back(static_cast<uint>(material), static_cast<uint>(material + 1));

	// Update flags
	m_Changed[MODELS] = true;
	m_Changed[MATERIALS] = true;

	m_UninitializedMeshes = true;

	// Return reference
	return geometry_ref(idx, *this);
}

instance_ref system::add_instance(const geometry_ref &geometry, glm::vec3 scaling, glm::vec3 translation, float degrees,
								  glm::vec3 axes)
{
	m_Changed[INSTANCES] = true;
	const size_t idx = m_Instances.size();
	m_InstanceChanged.push_back(true);

	instance_ref ref = instance_ref(idx, geometry, *this);
	ref.set_scaling(scaling);
	ref.set_rotation(degrees, axes);
	ref.set_translation(translation);

	m_Instances.push_back(ref);
	m_InstanceMatrices.push_back(ref.get_matrix());

	if (!m_ObjectLightIndices[geometry.get_index()].empty())
	{
		m_Changed[LIGHTS] = true;
		m_Changed[AREA_LIGHTS] = true;
	}
	return ref;
}

void system::update_instance(const instance_ref &instanceRef, const mat4 &transform)
{
	m_Changed[INSTANCES] = true;
	assert(m_Instances.size() > (size_t)instanceRef);

	m_InstanceMatrices[instanceRef.get_index()] = transform;
	m_InstanceChanged[instanceRef.get_index()] = true;

	if (!m_ObjectLightIndices[instanceRef.get_geometry_ref().get_index()].empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
}

void system::set_animation_to(const rfw::geometry_ref &instanceRef, float timeInSeconds)
{
#if ANIMATION_ENABLED
	const auto index = instanceRef.get_index();

	assert(m_Instances.size() > (size_t)instanceRef);
	m_Models[index]->set_time(timeInSeconds);
	m_ModelChanged[index] = true;

	if (!m_ObjectLightIndices[index].empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
#endif
}

rfw::HostMaterial rfw::system::get_material(size_t index) const { return m_Materials->get(uint(index)); }

void rfw::system::set_material(size_t index, const rfw::HostMaterial &mat)
{
	const bool wasEmissive = m_Materials->get_material_light_flags()[index];
	m_Changed[MATERIALS] = true;
	m_Materials->set(static_cast<uint>(index), mat);

	// Material was made emissive or used to be emissive, thus we need to update area lights
	if (mat.isEmissive() || wasEmissive)
	{
		for (size_t i = 0; i < m_ObjectMaterialRange.size(); i++)
		{
			const auto &[first, last] = m_ObjectMaterialRange[i];
			if (index >= first && index < last)
			{
				m_ObjectLightIndices[i] = m_Models[i]->get_light_indices(m_Materials->get_material_light_flags(), true);
				m_Changed[LIGHTS] = true;
				m_Changed[AREA_LIGHTS] = true;
			}
		}
	}
}

int system::add_material(const glm::vec3 &color, float roughness)
{
	HostMaterial mat{};
	mat.color = color;
	mat.roughness = roughness;
	m_Changed[MATERIALS] = true;
	const int index = static_cast<int>(m_Materials->add(mat));
	return index;
}

void rfw::system::render_frame(const Camera &camera, RenderStatus status, bool toneMap)
{
	assert(m_TargetID > 0);
	if (m_UpdateThread.valid())
		m_UpdateThread.get();

	if (m_ShouldReset)
		status = Reset;

	timer t = {};
	m_Context->render_frame(camera, status);

	if (toneMap)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferID);
		glDisable(GL_DEPTH_TEST);
		CheckGL();

		m_ToneMapShader.bind();
		m_ToneMapShader.set_uniform("tex", 0);
		m_ToneMapShader.set_uniform("params", vec4(camera.contrast, camera.brightness, 0, 0));
		CheckGL();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_TargetID);
		CheckGL();

		draw_quad();
		CheckGL();
		m_ToneMapShader.unbind();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	m_ShouldReset = false;
	m_RenderStat.add_sample(t.elapsed());
}

light_ref system::add_point_light(const glm::vec3 &position, const glm::vec3 &radiance)
{
	size_t index = m_PointLights.size();
	PointLight pl{};
	pl.position = position;
	pl.energy = sqrt(dot(radiance, radiance));
	pl.radiance = radiance;
	m_PointLights.push_back(pl);
	m_Changed[LIGHTS] = true;
	return light_ref(index, light_ref::POINT, *this);
}

light_ref system::add_spot_light(const glm::vec3 &position, float inner_deg, const glm::vec3 &radiance, float outer_deg,
								 const glm::vec3 &direction)
{
	size_t index = m_SpotLights.size();
	SpotLight sl{};
	sl.position = position;
	sl.cosInner = cos(radians(inner_deg));
	sl.radiance = radiance;
	sl.cosOuter = cos(radians(outer_deg));
	sl.direction = normalize(direction);
	sl.energy = sqrt(dot(radiance, radiance));
	m_SpotLights.push_back(sl);
	m_Changed[LIGHTS] = true;
	return light_ref(index, light_ref::SPOT, *this);
}

light_ref system::add_directional_light(const glm::vec3 &direction, const glm::vec3 &radiance)
{
	size_t index = m_DirectionalLights.size();
	DirectionalLight dl{};
	dl.direction = normalize(direction);
	dl.energy = sqrt(dot(radiance, radiance));
	dl.radiance = radiance;
	m_DirectionalLights.push_back(dl);
	m_Changed[LIGHTS] = true;
	return light_ref(index, light_ref::DIRECTIONAL, *this);
}

light_ref system::get_area_light_ref(size_t index)
{
	assert(index < m_AreaLights.size());
	if (index >= m_AreaLights.size())
		throw std::runtime_error("Requested point light index does not exist.");

	return rfw::light_ref(index, light_ref::AREA, *this);
}

light_ref system::get_point_light_ref(size_t index)
{
	assert(index < m_PointLights.size());
	if (index >= m_PointLights.size())
		throw std::runtime_error("Requested point light index does not exist.");

	return rfw::light_ref(index, light_ref::POINT, *this);
}
light_ref system::get_spot_light_ref(size_t index)
{
	assert(index < m_SpotLights.size());
	if (index >= m_SpotLights.size())
		throw std::runtime_error("Requested spot light index does not exist.");

	return rfw::light_ref(index, light_ref::SPOT, *this);
}

light_ref system::get_directional_light_ref(size_t index)
{
	assert(index < m_DirectionalLights.size());
	if (index >= m_DirectionalLights.size())
		throw std::runtime_error("Requested directional light index does not exist.");

	return rfw::light_ref(index, light_ref::DIRECTIONAL, *this);
}

void system::set_light_position(const light_ref &ref, const glm::vec3 &position)
{
	switch (ref.type)
	{
	case (light_ref::POINT):
		m_PointLights[ref.index].position = position;
		break;
	case (light_ref::DIRECTIONAL):
		return;
	case (light_ref::SPOT):
		m_SpotLights[ref.index].position = position;
		break;
	default:
		return;
	}
	m_Changed[LIGHTS] = true;
}

void system::set_light_radiance(const light_ref &ref, const glm::vec3 &radiance)
{
	switch (ref.type)
	{
	case (light_ref::POINT):
		m_PointLights[ref.index].radiance = radiance;
		break;
	case (light_ref::DIRECTIONAL):
		m_DirectionalLights[ref.index].radiance = radiance;
		return;
	case (light_ref::SPOT):
		m_SpotLights[ref.index].radiance = radiance;
		break;
	default:
		return;
	}
	m_Changed[LIGHTS] = true;
}

rfw::AvailableRenderSettings system::get_available_settings() const
{
	if (m_Context)
		return m_Context->get_settings();
	else
	{
		WARNING("Available settings requested while no context was loaded yet.");
		return {};
	}
}

void system::set_setting(const rfw::RenderSetting &setting) const
{
	if (m_Context)
		m_Context->set_setting(setting);
	else
		WARNING("Setting was set while no context was loaded yet.");
}

#if 0
AABB rfw::system::calculateSceneBounds() const
{
	AABB bounds;

	for (size_t i = 0, s = m_Instances.size(); i < s; i++)
	{
		const auto &instance = m_Instances.at(i);
		const auto &matrix = m_InstanceMatrices.at(i);
		const auto meshIdx = instance.get_geometry_ref().get_index();

		for (const auto &[index, mesh] : m_Models.at(meshIdx)->get_meshes())
		{
			for (uint v = 0; v < mesh.vertexCount; v++)
			{
				const vec3 transformedVertex = matrix * mesh.vertices[v];
				bounds.grow(transformedVertex);
			}
		}
	}

	return bounds;
}
#endif

void system::set_probe_index(glm::uvec2 pixelIdx) { m_Context->set_probe_index((m_ProbeIndex = pixelIdx)); }

glm::uvec2 rfw::system::get_probe_index() const { return m_ProbeIndex; }

system::ProbeResult system::get_probe_result()
{
	m_Context->get_probe_results(&m_ProbedInstance, &m_ProbedPrimitive, &m_ProbeDistance);
	const std::tuple<int, int, int> result = m_InverseInstanceMapping.at(m_ProbedInstance);
	const int instanceID = std::get<0>(result);
	const int objectID = std::get<1>(result);
	const int meshID = std::get<2>(result);
	const rfw::instance_ref &reference = m_Instances.at(instanceID);
	const rfw::geometry::SceneTriangles *object = m_Models[objectID];
	const auto &mesh = object->get_meshes()[meshID].second;
	auto *triangle = const_cast<Triangle *>(&mesh.triangles[m_ProbedPrimitive]);
	const auto materialID = triangle->material;
	return ProbeResult(reference, meshID, static_cast<int>(m_ProbedPrimitive), triangle, materialID, m_ProbeDistance);
}

rfw::instance_ref *system::get_mutable_instances(size_t *size)
{
	if (size)
		*size = m_Instances.size();
	return m_Instances.data();
}

size_t system::get_instance_count() const { return m_Instances.size(); }

std::vector<rfw::geometry_ref> system::get_geometry()
{
	std::vector<rfw::geometry_ref> geometry(m_Models.size(), rfw::geometry_ref(0, *this));
	for (size_t i = 0, s = m_Models.size(); i < s; i++)
		geometry.at(i).m_Index = i;
	return geometry;
}

size_t system::get_geometry_count() const { return m_Models.size(); }

rfw::RenderStats rfw::system::get_statistics() const
{
	auto stats = m_Context->get_stats();
	stats.animationTime = m_AnimationStat.get_average();
	stats.renderTime = m_RenderStat.get_average();
	return stats;
}

size_t rfw::system::request_mesh_index()
{
	if (m_EmptyMeshSlots > 0)
	{
		for (size_t i = 0, s = m_MeshSlots.size(); i < s; i++)
		{
			if (!m_MeshSlots[i])
			{
				m_MeshSlots[i] = true;
				m_EmptyMeshSlots--;
				return i;
			}
		}

		throw RfwException("m_EmptySlots does not adhere to actual available empty slots");
	}

	const size_t index = m_MeshSlots.size();
	m_MeshSlots.push_back(true);
	return index;
}

size_t rfw::system::request_instance_index()
{
	if (m_EmptyInstanceSlots > 0)
	{
		for (size_t i = 0, s = m_InstanceSlots.size(); i < s; i++)
		{
			if (!m_InstanceSlots[i])
			{
				m_InstanceSlots[i] = true;
				m_EmptyInstanceSlots--;
				return i;
			}
		}

		throw RfwException("m_EmptySlots does not adhere to actual available empty slots");
	}

	const size_t index = m_InstanceSlots.size();
	m_InstanceSlots.push_back(true);
	m_InverseInstanceMapping.emplace_back(0, 0, 0);
	return index;
}

void system::update_area_lights()
{
	m_AreaLights.clear();

	for (size_t i = 0; i < m_Instances.size(); i++)
	{
		const auto &reference = m_Instances[i];
		const auto &matrix = m_InstanceMatrices[i];
		const auto &geometry = reference.get_geometry_ref();

		const auto &lightIndices = geometry.get_light_indices();
		if (lightIndices.empty())
			continue;

		const auto &meshes = geometry.get_meshes();
		const auto &meshTransforms = geometry.get_mesh_matrices();

		for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
		{
			const auto &[meshSlot, mesh] = meshes[i];

			// We need a mutable reference to triangles to set their appropriate light triangle index
			auto *triangles = const_cast<Triangle *>(mesh.triangles);
			// Get appropriate light index vector
			const auto &meshLightIndices = lightIndices[i];
			const auto transform = meshTransforms[i] * matrix;
			const auto normal_transform = transform.inversed().transposed();

			// Generate arealights for current mesh
			for (const int index : meshLightIndices)
			{
				auto &triangle = triangles[index];
				const auto &material = m_Materials->get(triangle.material);

				assert(material.isEmissive());

				const vec4 lv0 = vec4(triangle.vertex0, 1.0f) * transform;
				const vec4 lv1 = vec4(triangle.vertex1, 1.0f) * transform;
				const vec4 lv2 = vec4(triangle.vertex2, 1.0f) * transform;

				const vec3 lN = normal_transform * vec4(triangle.Nx, triangle.Ny, triangle.Nz, 0);

				AreaLight light{};
				light.vertex0 = vec3(lv0);
				light.vertex1 = vec3(lv1);
				light.vertex2 = vec3(lv2);
				light.position = (light.vertex0 + light.vertex1 + light.vertex2) * (1.0f / 3.0f);
				light.energy = length(material.color);
				light.radiance = material.color;
				light.normal = lN;
				light.triIdx = index;
				light.instIdx = static_cast<uint>(i);
				light.normal = lN;

				triangle.lightTriIdx = static_cast<uint>(m_AreaLights.size());
				triangle.updateArea();

				light.area = triangle.area;
				m_AreaLights.push_back(light);
			}

			m_ModelChanged.at(reference.get_geometry_ref().get_index()) = false;
			m_Context->set_mesh(meshSlot, mesh);
		}
	}
}

utils::array_proxy<rfw::instance_ref> system::get_instances() const { return m_Instances; }