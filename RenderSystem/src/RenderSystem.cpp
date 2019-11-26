#include "RenderSystem.h"

#include "utils/gl/GLTexture.h"

#include "MaterialList.h"
#include "SceneTriangles.h"
#include "AssimpObject.h"
#include "gLTF/gLTFObject.h"

#include <ContextExport.h>
#include <future>

#include "utils/File.h"
#include "utils/Timer.h"
#include "Quad.h"

#include "utils/gl/CheckGL.h"

#ifdef WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <libloaderapi.h>
#include <utils/Timer.h>
#elif defined(__linux__) || defined(__APPLE__)
#include <dlfcn.h>
#include <utils/gl/CheckGL.h>
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

using namespace rfw;

rfw::RenderSystem::RenderSystem() { m_Materials = new MaterialList(); }

rfw::RenderSystem::~RenderSystem()
{
	if (m_Context)
		unloadRenderAPI();

	for (auto *object : m_Models)
		delete object;

	delete m_Materials;
	m_Materials = nullptr;
}

void *LoadModule(const std::string_view &file)
{
	void *module = nullptr;
#ifdef WIN32
	module = LoadLibrary(file.data());
	if (!module)
		fprintf(stderr, "Loading library \"%s\" error: %u\n", file.data(), GetLastError());
#else
	module = dlopen(file.data(), RTLD_NOW);
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

void *LoadModuleFunction(void *module, const std::string_view &funcName)
{
#ifdef WIN32
	return (void *)GetProcAddress((HMODULE)module, funcName.data());
#else
	return (void *)dlsym(module, funcName.data());
#endif
}

static_assert(sizeof(Material) == sizeof(DeviceMaterial), "Material structs are not same size.");
static_assert(sizeof(Triangle) == sizeof(DeviceTriangle), "Triangle structs are not same size.");
static_assert(sizeof(AreaLight) == sizeof(DeviceAreaLight), "Area light structs are not same size.");
static_assert(sizeof(PointLight) == sizeof(DevicePointLight), "Point light structs are not same size.");
static_assert(sizeof(SpotLight) == sizeof(DeviceSpotLight), "Spot light structs are not same size.");
static_assert(sizeof(DirectionalLight) == sizeof(DeviceDirectionalLight), "Directional light structs are not same size.");

void rfw::RenderSystem::loadRenderAPI(const std::string_view &name)
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
	const std::string libName = std::string(name) + std::string(extension);
	const std::string libPath = cwd + '/' + libName;

	if (m_Context)
	{
		throw std::runtime_error("A RenderContext was already loaded, unload current context before loading a new context.");
	}

	if (!utils::file::exists(libPath))
	{
		throw std::runtime_error("Library does not exist.");
	}

	m_ContextModule = LoadModule(libPath);
	if (!m_ContextModule)
	{
		const std::string message = std::string("Could not load library: ") + libPath;
		throw std::runtime_error(message);
	}

	void *createPtr = LoadModuleFunction(m_ContextModule, CREATE_RENDER_CONTEXT_FUNC_NAME);
	void *destroyPtr = LoadModuleFunction(m_ContextModule, DESTROY_RENDER_CONTEXT_FUNC_NAME);
	m_CreateContextFunction = (CreateContextFunction)createPtr;
	m_DestroyContextFunction = (DestroyContextFunction)destroyPtr;

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

void rfw::RenderSystem::unloadRenderAPI()
{
	if (m_Context)
	{
		m_Context->cleanup();
		m_DestroyContextFunction(m_Context);
	}
	if (m_ContextModule)
	{
#ifdef WIN32
		FreeLibrary((HMODULE)m_ContextModule);
#else
		dlclose(m_ContextModule);
#endif
	}

	m_Context = nullptr;
	m_ContextModule = nullptr;
}

void rfw::RenderSystem::setTarget(std::shared_ptr<utils::Window> window) { m_Context->init(window); }

void RenderSystem::setTarget(GLuint *textureID, uint width, uint height)
{
	m_Context->init(textureID, width, height);
	CheckGL();
}

void RenderSystem::setTarget(rfw::utils::GLTexture *texture)
{
	if (texture == nullptr)
		throw std::runtime_error("Invalid texture.");
	m_Context->init(&texture->m_ID, texture->getWidth(), texture->getHeight());
	CheckGL();
}

void RenderSystem::setSkybox(std::string_view filename)
{
	if (!utils::file::exists(filename))
	{
		WARNING("File: %s does not exist.", filename.data());
		throw std::runtime_error("Skybox file does not exist.");
	}

	m_Skybox = new Skybox(filename);
	m_Changed[SKYBOX] = true;
}

void rfw::RenderSystem::synchronize()
{
	rfw::utils::Timer t;

	if (m_Changed[SKYBOX])
		m_Context->setSkyDome(m_Skybox->getBuffer(), m_Skybox->getWidth(), m_Skybox->getHeight());

	if (m_Materials->isDirty())
		m_Materials->generateDeviceMaterials();

	// Update materials
	if (!m_Materials->getMaterials().empty() && m_Changed[MATERIALS])
	{
		const std::vector<TextureData> &textures = m_Materials->getTextureDescriptors();
		if (!textures.empty())
			m_Context->setTextures(textures);

		m_Context->setMaterials(m_Materials->getDeviceMaterials(), m_Materials->getMaterialTexIds());
	}

	// First update area lights as these might update models in the process.
	// Updating area lights first prevents us from having to update models twice
	if (m_Changed[AREA_LIGHTS] && !m_UninitializedMeshes)
	{
		updateAreaLights();
		m_Changed[LIGHTS] = true;
		m_Changed[AREA_LIGHTS] = false;
	}

	// Update loaded objects/models
	if (m_Changed[MODELS])
	{
		for (size_t i = 0; i < m_Models.size(); i++)
		{
			// Only update if changed
			if (!m_ModelChanged.at(i))
				continue;

			const auto &model = m_Models.at(i);
			const auto mesh = model->getMesh();

			assert(mesh.vertexCount > 0);
			assert(mesh.triangleCount > 0);
			assert(mesh.vertices);
			assert(mesh.normals);
			assert(mesh.triangles);

			m_Context->setMesh(i, mesh);

			// Reset state
			m_ModelChanged.at(i) = false;
		}
	}

	if (m_Changed[AREA_LIGHTS])
	{
		updateAreaLights();
		m_Changed[LIGHTS] = true;
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
			const auto &ref = m_Instances.at(i);
			const auto &matrix = m_InstanceMatrices.at(i);
			m_Context->setInstance(i, ref.getReference().getIndex(), matrix);
		}
	}

	if (m_Changed[LIGHTS])
	{
		LightCount count = {};
		count.areaLightCount = static_cast<uint>(m_AreaLights.size());
		count.pointLightCount = static_cast<uint>(m_PointLights.size());
		count.spotLightCount = static_cast<uint>(m_SpotLights.size());
		count.directionalLightCount = static_cast<uint>(m_DirectionalLights.size());
		m_Context->setLights(count, reinterpret_cast<const DeviceAreaLight *>(m_AreaLights.data()),
							 reinterpret_cast<const DevicePointLight *>(m_PointLights.data()), reinterpret_cast<const DeviceSpotLight *>(m_SpotLights.data()),
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

	//	utils::logger::log("Updated context in %3.3f", t.elapsed());
}

void RenderSystem::updateAnimationsTo(float timeInSeconds)
{
#if 0
	std::vector<std::future<void>> updates;

	for (size_t i = 0; i < m_Models.size(); i++)
	{
		auto object = m_Models.at(i);
		if (object->isAnimated())
		{
			m_ShouldReset = true;
			updates.push_back(std::async([this, i, object, timeInSeconds]() {
				object->transformTo(timeInSeconds);
				m_SetMeshMutex.lock();
				m_Context->setMesh(i, object->getMesh());
				m_SetMeshMutex.unlock();
			}));
		}
	}

	for (auto &update : updates)
		if (update.valid())
			update.get();
#else
	for (size_t i = 0; i < m_Models.size(); i++)
	{
		auto object = m_Models.at(i);
		if (object->isAnimated())
		{
			m_ShouldReset = true;
			object->transformTo(timeInSeconds);
			m_Context->setMesh(i, object->getMesh());
		}
	}
#endif
}

rfw::GeometryReference RenderSystem::getGeometryReference(size_t index)
{
	if (index >= m_Models.size())
		throw std::runtime_error("Geometry at given index does not exist.");

	return rfw::GeometryReference(index, *this);
}

rfw::InstanceReference RenderSystem::getInstanceReference(size_t index)
{
	if (index >= m_Instances.size())
		throw std::runtime_error("Instance at given index does not exist.");

	return m_Instances.at(index);
}

rfw::GeometryReference RenderSystem::addObject(std::string_view fileName, int material) { return addObject(fileName, false, glm::mat4(1.0f), material); }

rfw::GeometryReference RenderSystem::addObject(std::string_view fileName, bool normalize, int material)
{
	return addObject(fileName, normalize, glm::mat4(1.0f), material);
}

GeometryReference RenderSystem::addObject(std::string_view fileName, bool normalize, const glm::mat4 &preTransform, int material)
{
	if (!utils::file::exists(fileName))
		throw LoadException(std::string(fileName.data()));

	const size_t idx = m_Models.size();
	const size_t matFirst = m_Materials->getMaterials().size();

// Add model to list
#if USE_TINY_GLTF
	if (utils::string::ends_with(fileName, {".gltf", ".glb"}))
	{
		m_Models.emplace_back(new gLTFObject(fileName, m_Materials, static_cast<uint>(idx), preTransform, material));
	}
	else
#endif
	{
		m_Models.emplace_back(new AssimpObject(fileName, m_Materials, static_cast<uint>(idx), preTransform, normalize, material));
	}

	m_ModelChanged.push_back(true);

	const auto lightIndices = m_Models.at(idx)->getLightIndices(m_Materials->getMaterialLightFlags());
	if (!lightIndices.empty())
	{
		m_Changed[AREA_LIGHTS] = true;
		m_Changed[LIGHTS] = true;
	}
	m_ObjectLightIndices.push_back(lightIndices);

	m_ObjectMaterialRange.push_back(std::make_pair(static_cast<uint>(matFirst), static_cast<uint>(m_Materials->getMaterials().size())));

	// Update flags
	m_Changed[MODELS] = true;
	m_Changed[MATERIALS] = true;

	m_UninitializedMeshes = true;

	// Return reference
	return GeometryReference(idx, *this);
}

rfw::GeometryReference RenderSystem::addQuad(const glm::vec3 &N, const glm::vec3 &pos, float width, float height, const uint material)
{
	const size_t idx = m_Models.size();
	if (m_Materials->getMaterials().size() <= material)
		throw LoadException("Material does not exist.");

	// Add model to list
	m_Models.emplace_back(new Quad(N, pos, width, height, material));
	m_ModelChanged.push_back(true);
	m_ObjectLightIndices.push_back(m_Models.at(idx)->getLightIndices(m_Materials->getMaterialLightFlags()));
	m_ObjectMaterialRange.push_back(std::make_pair(material, material + 1));

	// Update flags
	m_Changed[MODELS] = true;

	m_UninitializedMeshes = true;

	// Return reference
	return GeometryReference(idx, *this);
}

InstanceReference RenderSystem::addInstance(const GeometryReference &geometry, glm::vec3 scaling, glm::vec3 translation, float degrees, glm::vec3 axes)
{
	m_Changed[INSTANCES] = true;
	const auto idx = m_Instances.size();
	const auto &object = m_Models.at(geometry.getIndex());
	m_InstanceChanged.push_back(true);
	InstanceReference ref = InstanceReference(idx, geometry, *this);
	ref.setScaling(scaling);
	ref.setRotation(degrees, axes);
	ref.setTranslation(translation);
	m_Instances.emplace_back(ref);
	m_InstanceMatrices.emplace_back(ref.getMatrix());

	if (!m_ObjectLightIndices.at(geometry.getIndex()).empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
	return ref;
}

void RenderSystem::updateInstance(const InstanceReference &instanceRef, const mat4 &transform)
{
	m_Changed[INSTANCES] = true;
	assert(m_Instances.size() > (size_t)instanceRef);
	m_InstanceMatrices.at(instanceRef.getIndex()) = transform;
	m_InstanceChanged.at(instanceRef.getIndex()) = true;

	if (!m_ObjectLightIndices.at(instanceRef.getReference().getIndex()).empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
}

void RenderSystem::setAnimationTime(const rfw::GeometryReference &instanceRef, float timeInSeconds)
{
	m_Changed[MODELS] = true;
#if ANIMATION_ENABLED
	assert(m_Instances.size() > (size_t)instanceRef);
	m_Models.at(instanceRef)->transformTo(timeInSeconds);
	m_ModelChanged.at(instanceRef) = true;
#endif

	if (!m_ObjectLightIndices.at(instanceRef.getIndex()).empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
}

rfw::HostMaterial rfw::RenderSystem::getMaterial(size_t index) { return m_Materials->get(uint(index)); }

void rfw::RenderSystem::setMaterial(size_t index, const rfw::HostMaterial &mat)
{
	// Do not change remove or add lights, this can be very expensive
	// TODO: Figure out how we could allow this
	if ((mat.isEmissive() && !m_Materials->get(static_cast<uint>(index)).isEmissive()) ||
		(!mat.isEmissive() && m_Materials->get(static_cast<uint>(index)).isEmissive()))
		return;

	// TODO: Figure out how we can efficiently change materials to be emissive
	m_Changed[MATERIALS] = true;
	m_Materials->set(uint(index), mat);
	if (mat.isEmissive())
	{
		for (size_t i = 0; i < m_ObjectMaterialRange.size(); i++)
		{
			const auto &range = m_ObjectMaterialRange.at(i);

			if (index >= range.first && index < range.second)
			{
				m_ObjectLightIndices.at(i) = m_Models.at(i)->getLightIndices(m_Materials->getMaterialLightFlags());
				m_Changed[LIGHTS] = true;
				m_Changed[AREA_LIGHTS] = true;
			}
		}
	}
}

uint RenderSystem::addMaterial(const glm::vec3 &color, float roughness)
{
	HostMaterial mat{};
	mat.color = color;
	mat.roughness = roughness;
	m_Changed[MATERIALS] = true;
	const uint index = m_Materials->add(mat);
	return index;
}

void rfw::RenderSystem::renderFrame(const Camera &camera, RenderStatus status)
{
	if (m_ShouldReset)
	{
		m_Context->update();
		status = Reset;
	}

	m_Context->renderFrame(camera, status);
	m_ShouldReset = false;
}

LightReference RenderSystem::addPointLight(const glm::vec3 &position, float energy, const glm::vec3 &radiance)
{
	size_t index = m_PointLights.size();
	PointLight pl{};
	pl.position = position;
	pl.energy = energy;
	pl.radiance = radiance;
	m_PointLights.push_back(pl);
	m_Changed[LIGHTS] = true;
	return LightReference(index, LightReference::POINT, *this);
}

LightReference RenderSystem::addSpotLight(const glm::vec3 &position, float cosInner, const glm::vec3 &radiance, float cosOuter, float energy,
										  const glm::vec3 &direction)
{
	size_t index = m_SpotLights.size();
	SpotLight sl{};
	sl.position = position;
	sl.cosInner = cosInner;
	sl.radiance = radiance;
	sl.cosOuter = cosOuter;
	sl.direction = normalize(direction);
	sl.energy = energy;
	m_SpotLights.push_back(sl);
	m_Changed[LIGHTS] = true;
	return LightReference(index, LightReference::SPOT, *this);
}

LightReference RenderSystem::addDirectionalLight(const glm::vec3 &direction, float energy, const glm::vec3 &radiance)
{
	size_t index = m_DirectionalLights.size();
	DirectionalLight dl{};
	dl.direction = normalize(direction);
	dl.energy = energy;
	dl.radiance = radiance;
	m_DirectionalLights.push_back(dl);
	m_Changed[LIGHTS] = true;
	return LightReference(index, LightReference::DIRECTIONAL, *this);
}

LightReference RenderSystem::getAreaLightReference(size_t index)
{
	assert(index < m_AreaLights.size());
	if (index >= m_AreaLights.size())
		throw std::runtime_error("Requested point light index does not exist.");

	return rfw::LightReference(index, LightReference::AREA, *this);
}

LightReference RenderSystem::getPointLightReference(size_t index)
{
	assert(index < m_PointLights.size());
	if (index >= m_PointLights.size())
		throw std::runtime_error("Requested point light index does not exist.");

	return rfw::LightReference(index, LightReference::POINT, *this);
}
LightReference RenderSystem::getSpotLightReference(size_t index)
{
	assert(index < m_SpotLights.size());
	if (index >= m_SpotLights.size())
		throw std::runtime_error("Requested spot light index does not exist.");

	return rfw::LightReference(index, LightReference::SPOT, *this);
}

LightReference RenderSystem::getDirectionalLightReference(size_t index)
{
	assert(index < m_DirectionalLights.size());
	if (index >= m_DirectionalLights.size())
		throw std::runtime_error("Requested directional light index does not exist.");

	return rfw::LightReference(index, LightReference::DIRECTIONAL, *this);
}

void RenderSystem::setPosition(const LightReference &ref, const glm::vec3 &position)
{
	switch (ref.m_Type)
	{
	case (LightReference::POINT):
		m_PointLights.at(ref.getIndex()).position = position;
		break;
	case (LightReference::DIRECTIONAL):
		return;
	case (LightReference::SPOT):
		m_SpotLights.at(ref.getIndex()).position = position;
		break;
	default:
		return;
	}
	m_Changed[LIGHTS] = true;
}

void RenderSystem::setRadiance(const LightReference &ref, const glm::vec3 &radiance)
{
	switch (ref.m_Type)
	{
	case (LightReference::POINT):
		m_PointLights.at(ref.getIndex()).radiance = radiance;
		break;
	case (LightReference::DIRECTIONAL):
		m_DirectionalLights.at(ref.getIndex()).radiance = radiance;
		return;
	case (LightReference::SPOT):
		m_SpotLights.at(ref.getIndex()).radiance = radiance;
		break;
	default:
		return;
	}
	m_Changed[LIGHTS] = true;
}

void RenderSystem::setEnergy(const LightReference &ref, float energy)
{
	switch (ref.m_Type)
	{
	case (LightReference::POINT):
		m_PointLights.at(ref.getIndex()).energy = energy;
		break;
	case (LightReference::DIRECTIONAL):
		m_DirectionalLights.at(ref.getIndex()).energy = energy;
		return;
	case (LightReference::SPOT):
		m_SpotLights.at(ref.getIndex()).energy = energy;
		break;
	default:
		return;
	}
	m_Changed[LIGHTS] = true;
}

rfw::AvailableRenderSettings RenderSystem::getAvailableSettings() const
{
	if (m_Context)
		return m_Context->getAvailableSettings();
	else
	{
		WARNING("Available settings requested while no context was loaded yet.");
		return {};
	}
}

void RenderSystem::setSetting(const rfw::RenderSetting &setting) const
{
	if (m_Context)
		m_Context->setSetting(setting);
	else
		WARNING("Setting was set while no context was loaded yet.");
}

AABB rfw::RenderSystem::calculateSceneBounds() const
{
	AABB bounds;

	for (size_t i = 0, s = m_Instances.size(); i < s; i++)
	{
		const auto &instance = m_Instances.at(i);
		const auto &matrix = m_InstanceMatrices.at(i);
		const auto meshIdx = instance.getReference().getIndex();
		const auto mesh = m_Models.at(meshIdx)->getMesh();

		for (uint v = 0; v < mesh.vertexCount; v++)
		{
			const vec3 transformedVertex = matrix * mesh.vertices[v];
			bounds.grow(transformedVertex);
		}
	}

	return bounds;
}

void RenderSystem::setProbeIndex(glm::uvec2 pixelIdx) { m_Context->setProbePos((m_ProbeIndex = pixelIdx)); }
glm::uvec2 rfw::RenderSystem::getProbeIndex() const { return m_ProbeIndex; }

RenderSystem::ProbeResult RenderSystem::getProbeResult()
{
	m_Context->getProbeResults(&m_ProbedInstance, &m_ProbedPrimitive, &m_ProbeDistance);
	const auto &reference = m_Instances.at(m_ProbedInstance);
	const size_t materialIdx = m_Models.at(reference.getReference().getIndex())->getMaterialForPrim(m_ProbedPrimitive);
	return ProbeResult(reference, m_ProbedPrimitive, materialIdx, m_ProbeDistance);
}

rfw::InstanceReference *RenderSystem::getMutableInstances() { return m_Instances.data(); }

size_t RenderSystem::getInstanceCount() const { return m_Instances.size(); }

std::vector<rfw::GeometryReference> RenderSystem::getGeometry()
{
	std::vector<rfw::GeometryReference> geometry(m_Models.size(), rfw::GeometryReference(0, *this));
	for (size_t i = 0, s = m_Models.size(); i < s; i++)
		geometry.at(i).m_Index = i;
	return geometry;
}

size_t RenderSystem::getGeometryCount() const { return m_Models.size(); }

rfw::RenderStats rfw::RenderSystem::getRenderStats() const { return m_Context->getStats(); }

void RenderSystem::updateAreaLights()
{
	m_AreaLights.clear();

	for (size_t i = 0; i < m_Instances.size(); i++)
	{
		const auto &reference = m_Instances.at(i);
		const auto &matrix = m_InstanceMatrices.at(i);
		const auto meshIdx = reference.getReference().getIndex();

		const auto &lightIndices = m_ObjectLightIndices.at(meshIdx);
		if (lightIndices.empty())
			continue;

		const mat3 nMatrix = mat3(matrix);

		Mesh mesh = m_Models.at(meshIdx)->getMesh();
		// We need a mutable reference to triangles to set their appropriate light triangle index
		auto *triangles = m_Models.at(meshIdx)->getTriangles();
		for (const uint index : lightIndices)
		{
			auto &triangle = triangles[index];
			const auto &material = m_Materials->get(triangle.material);

			assert(material.isEmissive());

			const vec4 lv0 = matrix * vec4(triangle.vertex0, 1.0f);
			const vec4 lv1 = matrix * vec4(triangle.vertex1, 1.0f);
			const vec4 lv2 = matrix * vec4(triangle.vertex2, 1.0f);

			const vec3 lN = nMatrix * vec3(triangle.Nx, triangle.Ny, triangle.Nz);

			AreaLight light{};
			light.vertex0 = vec3(lv0);
			light.vertex1 = vec3(lv1);
			light.vertex2 = vec3(lv2);
			light.position = (light.vertex0 + light.vertex1 + light.vertex2) * (1.0f / 3.0f);
			light.energy = 1.0f;
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

		m_ModelChanged.at(reference.getReference().getIndex()) = false;
		m_Context->setMesh(reference.getReference().getIndex(), mesh);
	}

	m_AreaLights.resize(m_AreaLights.size());
}

void InstanceReference::setTranslation(const glm::vec3 value) { m_Members->m_Translation = value; }

void InstanceReference::setRotation(const float degrees, const glm::vec3 axis)
{
	m_Members->m_Rotation = glm::rotate(glm::identity<glm::quat>(), radians(degrees), axis);
}
void InstanceReference::setRotation(const glm::quat &q) { m_Members->m_Rotation = q; }

void InstanceReference::setScaling(const glm::vec3 value) { m_Members->m_Scaling = value; }

void InstanceReference::translate(const glm::vec3 offset) { m_Members->m_Translation = offset; }

void InstanceReference::rotate(const float degrees, const glm::vec3 axis)
{
	m_Members->m_Rotation = glm::rotate(m_Members->m_Rotation, radians(degrees), axis);
}

void InstanceReference::scale(const glm::vec3 offset) { m_Members->m_Scaling = offset; }

void InstanceReference::update() const { m_Members->m_System->updateInstance(*this, getMatrix()); }

glm::mat4 InstanceReference::getMatrix() const
{
	const auto T = glm::translate(glm::mat4(1.0f), m_Members->m_Translation);
	const auto R = glm::mat4(m_Members->m_Rotation);
	const auto S = glm::scale(glm::mat4(1.0f), m_Members->m_Scaling);
	return T * R * S;
}

glm::mat3 InstanceReference::getInverseMatrix() const
{
	const auto T = glm::translate(glm::mat4(1.0f), m_Members->m_Translation);
	const auto R = glm::mat4(m_Members->m_Rotation);
	return inverse(mat3(T * R));
}

InstanceReference::Members::Members(const GeometryReference &ref) : m_Reference(ref) {}

void GeometryReference::setAnimationTime(const float time) const { m_System->setAnimationTime(*this, time); }

rfw::Mesh GeometryReference::getMesh() const { return m_System->m_Models.at(m_Index)->getMesh(); }