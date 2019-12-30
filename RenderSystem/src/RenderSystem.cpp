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
#include "utils/gl/GLDraw.h"
#include "utils/Concurrency.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <libloaderapi.h>
#include <utils/Timer.h>
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

rfw::RenderSystem::RenderSystem() : m_ThreadPool(std::thread::hardware_concurrency()), m_ToneMapShader("shaders/draw-tex.vert", "shaders/tone-map.frag")
{
	m_Materials = new MaterialList();
	glGenFramebuffers(1, &m_FrameBufferID);
	m_ToneMapShader.bind();
	m_ToneMapShader.setUniform("view", mat4(1.0f));
	m_ToneMapShader.unbind();

	m_TargetWidth = 0;
	m_TargetHeight = 0;
}

rfw::RenderSystem::~RenderSystem()
{
	m_ThreadPool.stop(true);

	if (m_Context)
		unloadRenderAPI();

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
	return static_cast<void *>(dlsym(module, funcName.data()));
#endif
}

static_assert(sizeof(Material) == sizeof(DeviceMaterial), "Material structs are not same size.");
static_assert(sizeof(Triangle) == sizeof(DeviceTriangle), "Triangle structs are not same size.");
static_assert(sizeof(AreaLight) == sizeof(DeviceAreaLight), "Area light structs are not same size.");
static_assert(sizeof(PointLight) == sizeof(DevicePointLight), "Point light structs are not same size.");
static_assert(sizeof(SpotLight) == sizeof(DeviceSpotLight), "Spot light structs are not same size.");
static_assert(sizeof(DirectionalLight) == sizeof(DeviceDirectionalLight), "Directional light structs are not same size.");

void rfw::RenderSystem::loadRenderAPI(std::string name)
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

	m_ContextModule = LoadModule(libPath.data());
	if (!m_ContextModule)
	{
		const std::string message = std::string("Could not load library: ") + libPath;
		throw std::runtime_error(std::move(message));
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
		FreeLibrary(HMODULE(m_ContextModule));
#else
		dlclose(m_ContextModule);
#endif
	}

	m_Context = nullptr;
	m_ContextModule = nullptr;
}

void RenderSystem::setTarget(GLuint *textureID, uint width, uint height)
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

void RenderSystem::setTarget(rfw::utils::GLTexture *texture)
{
	assert(texture != nullptr);

	if (texture == nullptr)
		throw std::runtime_error("Invalid texture.");
	m_Context->init(&texture->m_ID, texture->getWidth(), texture->getHeight());

	m_TargetID = texture->m_ID;
	m_TargetWidth = texture->getWidth();
	m_TargetHeight = texture->getHeight();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CheckGL();
}

void RenderSystem::setSkybox(std::string filename)
{

	if (m_Skybox.getSource() == std::string(filename))
		return;

	if (!file::exists(filename))
	{
		WARNING("File: %s does not exist.", filename.data());
		throw std::runtime_error("Skybox file does not exist.");
	}

	m_Skybox.load(filename);
	m_Changed[SKYBOX] = true;
}

void RenderSystem::synchronize()
{
	if (m_AnimationsThread.valid())
		m_AnimationsThread.get();

	if (m_Changed[SKYBOX])
		m_Context->setSkyDome(m_Skybox.getBuffer(), m_Skybox.getWidth(), m_Skybox.getHeight());

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

	if (m_Changed[ANIMATED])
	{
		for (SceneTriangles *object : m_Models)
		{
			if (!object->isAnimated())
				continue;

			const auto changedMeshes = object->getChangedMeshes();
			auto meshes = object->getMeshes();

			for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
			{
				if (!changedMeshes[i])
					continue;
				const auto &[index, mesh] = meshes[i];
				m_Context->setMesh(index, mesh);
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

			for (const auto &[meshSlot, mesh] : m_Models[i]->getMeshes())
			{
				assert(mesh.vertexCount > 0);
				assert(mesh.triangleCount > 0);
				assert(mesh.vertices);
				assert(mesh.normals);
				assert(mesh.triangles);

				m_Context->setMesh(meshSlot, mesh);
			}

			// Reset state
			m_ModelChanged[i] = false;
		}
	}

	if (m_Changed[AREA_LIGHTS])
	{
		updateAreaLights();
		m_Changed[LIGHTS] = true;
	}

	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		rfw::InstanceReference &instRef = m_Instances[i];

		if (m_InstanceChanged[i]) // Need to update this instance anyway
		{
			// Update instance
			const auto &matrix = m_InstanceMatrices[i];
			const auto geometry = instRef.getGeometryReference();

			const auto &instanceMapping = instRef.getIndices();
			const auto &meshes = geometry.getMeshes();
			const auto &matrices = geometry.getMeshMatrices();

			for (size_t i = 0, s = meshes.size(); i < s; i++)
			{
				const auto meshID = meshes[i].first;
				const auto instanceID = instanceMapping[i];
				const mat4 transform = matrix * matrices[i];
				const mat3 inverse_transform = transpose(inverse(transform));
				m_Context->setInstance(instanceID, meshID, transform, inverse_transform);
			}

			m_InstanceChanged[i] = false;
		}
		else
		{
			const auto geometry = instRef.getGeometryReference();
			if (!geometry.isAnimated())
				continue;

			const auto object = instRef.getGeometryReference().getObject();
			const auto changedTransforms = object->getChangedMeshMatrices();

			// Update instance
			const auto &matrix = m_InstanceMatrices[i];

			const auto &instanceMapping = instRef.getIndices();
			const auto &meshes = geometry.getMeshes();
			const auto &matrices = geometry.getMeshMatrices();

			for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
			{
				if (!changedTransforms[i])
					continue;

				const auto meshID = meshes[i].first;
				const auto instanceID = instanceMapping[i];
				const mat4 transform = matrix * matrices[i];
				const mat3 inverse_transform = transpose(inverse(transform));
				m_Context->setInstance(instanceID, meshID, transform, inverse_transform);
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
			const auto &meshes = ref.getGeometryReference().getMeshes();
			const auto &matrices = ref.getGeometryReference().getMeshMatrices();

			for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
			{
				const auto meshID = meshes[i].first;
				const auto instanceID = instanceMapping[i];
				const mat4 transform = matrix * matrices[i];
				const mat3 inverse_transform = transpose(inverse(transform));
				m_Context->setInstance(instanceID, meshID, transform, inverse_transform);
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
}

void RenderSystem::updateAnimationsTo(const float timeInSeconds)
{
	Timer t = {};
#if ENABLE_THREADING
	std::vector<std::future<void>> updates;
	updates.reserve(m_Models.size());

	for (SceneTriangles *object : m_Models)
	{
		if (object->isAnimated())
			updates.push_back(m_ThreadPool.push([object, timeInSeconds](int) { object->transformTo(timeInSeconds); }));
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
		if (object->isAnimated())
		{
			m_Changed[ANIMATED] = true;
			m_ShouldReset = true;
			object->transformTo(timeInSeconds);
		}
	}
#endif
	m_AnimationStat.addSample(t.elapsed());
}

GeometryReference RenderSystem::getGeometryReference(size_t index)
{
	if (index >= m_Models.size())
		throw std::runtime_error("Geometry at given index does not exist.");

	return rfw::GeometryReference(index, *this);
}

InstanceReference RenderSystem::getInstanceReference(size_t index)
{
	if (index >= m_Instances.size())
		throw std::runtime_error("Instance at given index does not exist.");

	return m_Instances[index];
}

GeometryReference RenderSystem::addObject(std::string fileName, int material) { return addObject(std::move(fileName), false, glm::mat4(1.0f), material); }

GeometryReference RenderSystem::addObject(std::string fileName, bool normalize, int material)
{
	return addObject(std::move(fileName), normalize, glm::mat4(1.0f), material);
}

GeometryReference RenderSystem::addObject(std::string fileName, bool normalize, const glm::mat4 &preTransform, int material)
{
	if (!utils::file::exists(fileName))
		throw LoadException(fileName.data());

	const size_t idx = m_Models.size();
	const size_t matFirst = m_Materials->getMaterials().size();

	// Add model to list
	SceneTriangles *triangles = nullptr;

#if USE_TINY_GLTF
	if (utils::string::ends_with(fileName, {".gltf", ".glb"}))
		triangles = new gLTFObject(fileName, m_Materials, static_cast<uint>(idx), preTransform, material);
	else
#endif
		triangles = new AssimpObject(fileName, m_Materials, static_cast<uint>(idx), preTransform, normalize, material);

	triangles->prepareMeshes(*this);
	assert(!triangles->getMeshes().empty());

	m_Models.push_back(triangles);
	m_ModelChanged.push_back(true);

	const auto lightFlags = m_Materials->getMaterialLightFlags();

	const auto lightIndices = m_Models[idx]->getLightIndices(lightFlags, true);
	assert(lightIndices.size() == m_Models[idx]->getMeshes().size());

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
		m_ObjectMaterialRange.emplace_back(static_cast<uint>(matFirst), static_cast<uint>(m_Materials->getMaterials().size()));
	else
		m_ObjectMaterialRange.emplace_back(static_cast<uint>(material), static_cast<uint>(material + 1));

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
	SceneTriangles *triangles = new Quad(N, pos, width, height, material);

	// Update flags
	m_Changed[MODELS] = true;
	m_UninitializedMeshes = true;

	m_Models.push_back(triangles);
	m_ModelChanged.push_back(true);
	triangles->prepareMeshes(*this);

	const auto lightFlags = m_Materials->getMaterialLightFlags();
	const auto lightIndices = m_Models[idx]->getLightIndices(lightFlags, true);
	assert(lightIndices.size() == m_Models[idx]->getMeshes().size());

	for (const auto &mlIndices : lightIndices)
	{
		if (!mlIndices.empty())
		{
			m_Changed[AREA_LIGHTS] = true;
			m_Changed[LIGHTS] = true;
		}
	}

	m_ObjectLightIndices.push_back(lightIndices);
	m_ObjectMaterialRange.emplace_back(material, material + 1);

	// Update flags
	m_Changed[MODELS] = true;
	m_Changed[MATERIALS] = true;

	m_UninitializedMeshes = true;

	// Return reference
	return GeometryReference(idx, *this);
}

InstanceReference RenderSystem::addInstance(const GeometryReference &geometry, glm::vec3 scaling, glm::vec3 translation, float degrees, glm::vec3 axes)
{
	m_Changed[INSTANCES] = true;
	const size_t idx = m_Instances.size();
	const SceneTriangles *object = m_Models[geometry.getIndex()];
	m_InstanceChanged.push_back(true);

	InstanceReference ref = InstanceReference(idx, geometry, *this);
	ref.setScaling(scaling);
	ref.setRotation(degrees, axes);
	ref.setTranslation(translation);

	m_Instances.emplace_back(ref);
	m_InstanceMatrices.emplace_back(ref.getMatrix());

	if (!m_ObjectLightIndices[geometry.getIndex()].empty())
	{
		m_Changed[LIGHTS] = true;
		m_Changed[AREA_LIGHTS] = true;
	}
	return ref;
}

void RenderSystem::updateInstance(const InstanceReference &instanceRef, const mat4 &transform)
{
	m_Changed[INSTANCES] = true;
	assert(m_Instances.size() > (size_t)instanceRef);

	m_InstanceMatrices[instanceRef.getIndex()] = transform;
	m_InstanceChanged[instanceRef.getIndex()] = true;

	if (!m_ObjectLightIndices[instanceRef.getGeometryReference().getIndex()].empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
}

void RenderSystem::setAnimationTime(const rfw::GeometryReference &instanceRef, float timeInSeconds)
{
#if ANIMATION_ENABLED
	const auto index = instanceRef.getIndex();

	assert(m_Instances.size() > (size_t)instanceRef);
	m_Models[index]->transformTo(timeInSeconds);
	m_ModelChanged[index] = true;

	if (!m_ObjectLightIndices[index].empty())
		m_Changed[LIGHTS] = m_Changed[AREA_LIGHTS] = true;
#endif
}

rfw::HostMaterial rfw::RenderSystem::getMaterial(size_t index) const { return m_Materials->get(uint(index)); }

void rfw::RenderSystem::setMaterial(size_t index, const rfw::HostMaterial &mat)
{
	const bool wasEmissive = m_Materials->getMaterialLightFlags()[index];
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
				m_ObjectLightIndices[i] = m_Models[i]->getLightIndices(m_Materials->getMaterialLightFlags(), true);
				m_Changed[LIGHTS] = true;
				m_Changed[AREA_LIGHTS] = true;
			}
		}
	}
}

int RenderSystem::addMaterial(const glm::vec3 &color, float roughness)
{
	HostMaterial mat{};
	mat.color = color;
	mat.roughness = roughness;
	m_Changed[MATERIALS] = true;
	const int index = static_cast<int>(m_Materials->add(mat));
	return index;
}

void rfw::RenderSystem::renderFrame(const Camera &camera, RenderStatus status, bool toneMap)
{
	assert(m_TargetID > 0);
	if (m_UpdateThread.valid())
		m_UpdateThread.get();

	if (m_ShouldReset)
		status = Reset;

	Timer t = {};
	m_Context->renderFrame(camera, status);

	if (toneMap)
	{
		glBindFramebuffer(GL_FRAMEBUFFER, m_FrameBufferID);
		glDisable(GL_DEPTH_TEST);
		CheckGL();

		m_ToneMapShader.bind();
		m_ToneMapShader.setUniform("tex", 0);
		m_ToneMapShader.setUniform("params", vec4(camera.contrast, camera.brightness, 0, 0));
		CheckGL();

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, m_TargetID);
		CheckGL();

		drawQuad();
		CheckGL();
		m_ToneMapShader.unbind();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
	}

	m_ShouldReset = false;
	m_RenderStat.addSample(t.elapsed());
}

LightReference RenderSystem::addPointLight(const glm::vec3 &position, const glm::vec3 &radiance)
{
	size_t index = m_PointLights.size();
	PointLight pl{};
	pl.position = position;
	pl.energy = sqrt(dot(radiance, radiance));
	pl.radiance = radiance;
	m_PointLights.push_back(pl);
	m_Changed[LIGHTS] = true;
	return LightReference(index, LightReference::POINT, *this);
}

LightReference RenderSystem::addSpotLight(const glm::vec3 &position, float cosInner, const glm::vec3 &radiance, float cosOuter, const glm::vec3 &direction)
{
	size_t index = m_SpotLights.size();
	SpotLight sl{};
	sl.position = position;
	sl.cosInner = cosInner;
	sl.radiance = radiance;
	sl.cosOuter = cosOuter;
	sl.direction = normalize(direction);
	sl.energy = sqrt(dot(radiance, radiance));
	m_SpotLights.push_back(sl);
	m_Changed[LIGHTS] = true;
	return LightReference(index, LightReference::SPOT, *this);
}

LightReference RenderSystem::addDirectionalLight(const glm::vec3 &direction, const glm::vec3 &radiance)
{
	size_t index = m_DirectionalLights.size();
	DirectionalLight dl{};
	dl.direction = normalize(direction);
	dl.energy = sqrt(dot(radiance, radiance));
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
		const auto meshIdx = instance.getGeometryReference().getIndex();

		for (const auto &[index, mesh] : m_Models.at(meshIdx)->getMeshes())
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

void RenderSystem::setProbeIndex(glm::uvec2 pixelIdx) { m_Context->setProbePos((m_ProbeIndex = pixelIdx)); }
glm::uvec2 rfw::RenderSystem::getProbeIndex() const { return m_ProbeIndex; }

RenderSystem::ProbeResult RenderSystem::getProbeResult()
{
	m_Context->getProbeResults(&m_ProbedInstance, &m_ProbedPrimitive, &m_ProbeDistance);
	const std::tuple<int, int, int> result = m_InverseInstanceMapping[m_ProbedInstance];
	const int instanceID = std::get<0>(result);
	const int objectID = std::get<1>(result);
	const int meshID = std::get<2>(result);
	const rfw::InstanceReference &reference = m_Instances.at(instanceID);
	const SceneTriangles *object = m_Models[objectID];
	const auto &mesh = object->getMeshes()[meshID].second;
	Triangle *triangle = const_cast<Triangle *>(&mesh.triangles[m_ProbedPrimitive]);
	const auto materialID = triangle->material;
	return ProbeResult(reference, meshID, m_ProbedPrimitive, triangle, materialID, m_ProbeDistance);
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

rfw::RenderStats rfw::RenderSystem::getRenderStats() const
{
	auto stats = m_Context->getStats();
	stats.animationTime = m_AnimationStat.getAverage();
	stats.renderTime = m_RenderStat.getAverage();
	return stats;
}

size_t rfw::RenderSystem::requestMeshIndex()
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

size_t rfw::RenderSystem::requestInstanceIndex()
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

void RenderSystem::updateAreaLights()
{
	m_AreaLights.clear();

	for (size_t i = 0; i < m_Instances.size(); i++)
	{
		const auto &reference = m_Instances[i];
		const auto &matrix = m_InstanceMatrices[i];
		const auto &geometry = reference.getGeometryReference();

		const auto &lightIndices = geometry.getLightIndices();
		if (lightIndices.empty())
			continue;

		const auto &meshes = geometry.getMeshes();
		const auto &meshTransforms = geometry.getMeshMatrices();

		for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
		{
			const auto &[meshSlot, mesh] = meshes[i];

			// We need a mutable reference to triangles to set their appropriate light triangle index
			auto *triangles = const_cast<Triangle *>(mesh.triangles);
			// Get appropriate light index vector
			const auto &meshLightIndices = lightIndices[i];
			const auto transform = meshTransforms[i] * matrix;

			// Generate arealights for current mesh
			for (const int index : meshLightIndices)
			{
				auto &triangle = triangles[index];
				const auto &material = m_Materials->get(triangle.material);

				assert(material.isEmissive());

				const vec4 lv0 = transform * vec4(triangle.vertex0, 1.0f);
				const vec4 lv1 = transform * vec4(triangle.vertex1, 1.0f);
				const vec4 lv2 = transform * vec4(triangle.vertex2, 1.0f);

				const vec3 lN = transform * vec4(triangle.Nx, triangle.Ny, triangle.Nz, 0);

				AreaLight light{};
				light.vertex0 = vec3(lv0);
				light.vertex1 = vec3(lv1);
				light.vertex2 = vec3(lv2);
				light.position = (light.vertex0 + light.vertex1 + light.vertex2) * (1.0f / 3.0f);
				light.energy = sqrt(dot(material.color, material.color));
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

			m_ModelChanged.at(reference.getGeometryReference().getIndex()) = false;
			m_Context->setMesh(meshSlot, mesh);
		}
	}
}
const std::vector<rfw::InstanceReference> &RenderSystem::getInstances() const { return m_Instances; }

rfw::InstanceReference::InstanceReference(size_t index, GeometryReference reference, rfw::RenderSystem &system)
{
	m_Members = std::make_shared<Members>(reference);
	m_Members->index = index;
	m_Members->geomReference = reference;
	m_Members->rSystem = &system;
	assert(m_Members->rSystem);
	m_Members->translation = glm::vec3(0.0f);
	m_Members->rotation = glm::identity<glm::quat>();
	m_Members->scaling = glm::vec3(1.0f);

	const auto &meshes = reference.getMeshes();
	m_Members->instanceIDs.resize(meshes.size());
	for (int i = 0, s = static_cast<int>(meshes.size()); i < s; i++)
	{
		const int instanceID = static_cast<int>(system.requestInstanceIndex());
		system.m_InverseInstanceMapping[instanceID] = std::make_tuple(static_cast<int>(index), static_cast<int>(reference.getIndex()), i);
		m_Members->instanceIDs[i] = instanceID;
	}
}

void InstanceReference::setTranslation(const glm::vec3 value) { m_Members->translation = value; }

void InstanceReference::setRotation(const float degrees, const glm::vec3 axis)
{
	m_Members->rotation = glm::rotate(glm::identity<glm::quat>(), radians(degrees), axis);
}
void InstanceReference::setRotation(const glm::quat &q) { m_Members->rotation = q; }

void InstanceReference::setRotation(const glm::vec3 &euler) { m_Members->rotation = glm::quat(euler); }

void InstanceReference::setScaling(const glm::vec3 value) { m_Members->scaling = value; }

void InstanceReference::translate(const glm::vec3 offset) { m_Members->translation = offset; }

void InstanceReference::rotate(const float degrees, const glm::vec3 axis) { m_Members->rotation = glm::rotate(m_Members->rotation, radians(degrees), axis); }

void InstanceReference::scale(const glm::vec3 offset) { m_Members->scaling = offset; }

void InstanceReference::update() const { m_Members->rSystem->updateInstance(*this, getMatrix()); }

glm::mat4 InstanceReference::getMatrix() const
{
	const auto T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const auto R = glm::mat4(m_Members->rotation);
	const auto S = glm::scale(glm::mat4(1.0f), m_Members->scaling);
	return T * R * S;
}

glm::mat3 InstanceReference::getInverseMatrix() const
{
	const auto T = glm::translate(glm::mat4(1.0f), m_Members->translation);
	const auto R = glm::mat4(m_Members->rotation);
	return inverse(mat3(T * R));
}

InstanceReference::Members::Members(const GeometryReference &ref) : geomReference(ref) {}

bool rfw::GeometryReference::isAnimated() const { return m_System->m_Models[m_Index]->isAnimated(); }

void GeometryReference::setAnimationTime(const float time) const { m_System->setAnimationTime(*this, time); }

const std::vector<std::pair<size_t, rfw::Mesh>> &GeometryReference::getMeshes() const { return m_System->m_Models[m_Index]->getMeshes(); }

const std::vector<glm::mat4> &rfw::GeometryReference::getMeshMatrices() const { return m_System->m_Models[m_Index]->getMeshTransforms(); }

const std::vector<std::vector<int>> &rfw::GeometryReference::getLightIndices() const
{
	const auto lightFlags = m_System->m_Materials->getMaterialLightFlags();
	return m_System->m_Models[m_Index]->getLightIndices(lightFlags, false);
}

SceneTriangles *rfw::GeometryReference::getObject() const { return m_System->m_Models[m_Index]; }