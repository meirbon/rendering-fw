#include "PCH.h"
#include "Context.h"

using namespace cl;
using namespace rfw;

rfw::RenderContext *createRenderContext() { return new RTContext(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

RTContext::RTContext()
{
	m_Context = std::make_shared<CLContext>();
	m_DebugKernel = new CLKernel(m_Context, "clkernels/debug.cl", "draw", {512, 512}, {16, 16});
	m_RayGenKernel = new CLKernel(m_Context, "clkernels/ray_gen.cl", "generate_rays", {512}, {64});
	m_IntersectKernel = new CLKernel(m_Context, "clkernels/intersect.cl", "intersect_rays", {512}, {64});
	m_ShadeKernel = new CLKernel(m_Context, "clkernels/shade.cl", "shade_rays", {512}, {64});
	m_FinalizeKernel = new CLKernel(m_Context, "clkernels/finalize.cl", "finalize", {512, 512}, {16, 16});

	m_Skybox = new CLBuffer<glm::vec4>(m_Context, 1);
	const auto blueNoise = createBlueNoiseBuffer();
	m_BlueNoise = new CLBuffer<uint>(m_Context, blueNoise.size(), blueNoise.data());

	m_Camera = new cl::CLBuffer<CLCamera>(m_Context, 1);
	m_Counters = new cl::CLBuffer<uint>(m_Context, 4);

	m_PCs = new cl::CLBuffer<CLPotentialContribution>(m_Context, 1);
	m_Origins = new cl::CLBuffer<float4>(m_Context, 1);
	m_Directions = new cl::CLBuffer<float4>(m_Context, 1);
	m_States = new cl::CLBuffer<float4>(m_Context, 1);
	m_Accumulator = new cl::CLBuffer<float4>(m_Context, 1);

	m_AreaLights = new CLBuffer<DeviceAreaLight>(m_Context, 1);
	m_PointLights = new CLBuffer<DevicePointLight>(m_Context, 1);
	m_SpotLights = new CLBuffer<DeviceSpotLight>(m_Context, 1);
	m_DirLights = new CLBuffer<DeviceDirectionalLight>(m_Context, 1);

	m_Materials = new cl::CLBuffer<CLMaterial>(m_Context, 1);
	m_UintTextures = new cl::CLBuffer<uint>(m_Context, 1);
	m_FloatTextures = new cl::CLBuffer<float4>(m_Context, 1);
	m_Accumulator = new cl::CLBuffer<float4>(m_Context, 1);

	setCounters();
	setAccumulator();
	setSkybox();
	setLights();
	setBlueNoise();
	setCamera();
	setPCs();
	setOrigins();
	setDirections();
	setStates();
	setCounters();
	setMaterials();
	setTextures();
	setStride();
	setPathLength(0);
	setPathCount(0);
}

RTContext::~RTContext()
{
	delete m_Target;

	delete m_DebugKernel;
	delete m_RayGenKernel;
	delete m_IntersectKernel;
	delete m_ShadeKernel;
	delete m_FinalizeKernel;

	delete m_Skybox;

	delete m_AreaLights;
	delete m_PointLights;
	delete m_SpotLights;
	delete m_DirLights;

	delete m_BlueNoise;
	delete m_Camera;

	delete m_PCs;
	delete m_Origins;
	delete m_Directions;
	delete m_States;
	delete m_Accumulator;

	delete m_Counters;

	delete m_Materials;
	delete m_UintTextures;
	delete m_FloatTextures;
}

std::vector<rfw::RenderTarget> RTContext::getSupportedTargets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void RTContext::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void RTContext::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		auto error = glewInit();
		if (error != GLEW_NO_ERROR)
			throw std::runtime_error("Could not init GLEW.");
		m_InitializedGlew = true;
		CheckGL();
	}

	m_TargetID = *glTextureID;

	delete m_Target;
	m_Target = new cl::CLBuffer<glm::vec4, BufferType::TARGET>(m_Context, m_TargetID, width, height);

	m_DebugKernel->set_buffer(0, m_Target);
	m_FinalizeKernel->set_buffer(0, m_Target);

	m_FinalizeKernel->set_argument(2, width);
	m_FinalizeKernel->set_argument(3, width);

	m_DebugKernel->set_work_size({width, height});
	m_RayGenKernel->set_work_size({width * height});
	m_IntersectKernel->set_work_size({width * height});
	m_ShadeKernel->set_work_size({width * height});
	m_FinalizeKernel->set_work_size({width, height});

	m_Width = width;
	m_Height = height;

	resize_buffers();
}

void RTContext::cleanup() {}

void RTContext::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	const auto view = camera.getView();
	const auto cam = m_Camera->host_data();

	const vec3 right = view.p2 - view.p1;
	const vec3 up = view.p3 - view.p1;
	const vec4 posLensSize = vec4(view.pos, view.aperture);
	//	cam->right_spreadAngle = vec4(right, view.spreadAngle);
	//	cam->up = vec4(up, 0.0f);
	//	cam->p1 = vec4(view.p1, 0.0f);

	m_RayGenKernel->set_argument(5, posLensSize);
	m_RayGenKernel->set_argument(6, vec4(view.p1, 1.0f));
	m_RayGenKernel->set_argument(7, vec4(right, view.spreadAngle));
	m_RayGenKernel->set_argument(8, vec4(up, 1.0f));
	m_RayGenKernel->set_argument(9, m_Width);
	m_RayGenKernel->set_argument(10, m_Height);

	cam->samplesTaken = m_SampleIndex;
	cam->geometryEpsilon = 1e-5f;
	cam->scrwidth = m_Width;
	cam->scrheight = m_Height;
	m_Camera->copy_to_device(false);

	if (status == Reset)
	{
		m_SampleIndex = 0;
		m_Accumulator->clear(0);
	}

	setPhase(STAGE_PRIMARY_RAY);
	setPathLength(0);
	setPathCount(m_Width * m_Height);
	m_Counters->clear(0);

	m_Context->finish();
	m_RayGenKernel->set_work_size({m_Width * m_Height});
	m_RayGenKernel->run();
	m_Context->finish();

	m_IntersectKernel->set_work_size({m_Width * m_Height});
	m_IntersectKernel->run();
	m_Context->finish();

	m_ShadeKernel->set_work_size({m_Width * m_Height});
	m_ShadeKernel->run();
	m_Context->finish();

	glFinish();
	m_SampleIndex++;
	m_FinalizeKernel->set_work_size({m_Width, m_Height});
	m_FinalizeKernel->set_argument(2, m_Width);
	m_FinalizeKernel->set_argument(3, m_Height);
	m_FinalizeKernel->set_argument(4, 1.0f / float(m_SampleIndex));
	m_FinalizeKernel->run(*m_Target);
	m_Context->finish();
}

void RTContext::setMaterials(const std::vector<rfw::DeviceMaterial> &materials,
							 const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
	std::vector<rfw::DeviceMaterial> mats(materials.size());
	memcpy(mats.data(), materials.data(), materials.size() * sizeof(rfw::Material));

	for (size_t i = 0; i < materials.size(); i++)
	{
		auto &mat = reinterpret_cast<Material &>(mats.at(i));
		const MaterialTexIds &ids = texDescriptors[i];
		if (ids.texture[0] != -1)
			mat.texaddr0 = m_TexDescriptors[ids.texture[0]].texAddr;
		if (ids.texture[1] != -1)
			mat.texaddr1 = m_TexDescriptors[ids.texture[1]].texAddr;
		if (ids.texture[2] != -1)
			mat.texaddr2 = m_TexDescriptors[ids.texture[2]].texAddr;
		if (ids.texture[3] != -1)
			mat.nmapaddr0 = m_TexDescriptors[ids.texture[3]].texAddr;
		if (ids.texture[4] != -1)
			mat.nmapaddr1 = m_TexDescriptors[ids.texture[4]].texAddr;
		if (ids.texture[5] != -1)
			mat.nmapaddr2 = m_TexDescriptors[ids.texture[5]].texAddr;
		if (ids.texture[6] != -1)
			mat.smapaddr = m_TexDescriptors[ids.texture[6]].texAddr;
		if (ids.texture[7] != -1)
			mat.rmapaddr = m_TexDescriptors[ids.texture[7]].texAddr;
		if (ids.texture[9] != -1)
			mat.cmapaddr = m_TexDescriptors[ids.texture[9]].texAddr;
		if (ids.texture[10] != -1)
			mat.amapaddr = m_TexDescriptors[ids.texture[10]].texAddr;
	}

	if (!m_Materials || m_Materials->size() < materials.size())
	{
		delete m_Materials;
		m_Materials = new cl::CLBuffer<CLMaterial>(m_Context, mats.size());
	}

	memcpy(m_Materials->host_data(), mats.data(), mats.size() * sizeof(DeviceMaterial));
	m_Materials->copy_to_device(false);
	setMaterials();
}

void RTContext::setTextures(const std::vector<rfw::TextureData> &textures)
{
	m_TexDescriptors = textures;

	delete m_FloatTextures;
	delete m_UintTextures;

	size_t uintTexelCount = 0;
	size_t floatTexelCount = 0;

	std::vector<glm::vec4> floatTexs;
	std::vector<uint> uintTexs;

	for (const auto &tex : textures)
	{
		switch (tex.type)
		{
		case (TextureData::FLOAT4):
			floatTexelCount += tex.texelCount;
			break;
		case (TextureData::UINT):
			uintTexelCount += tex.texelCount;
			break;
		}
	}

	floatTexs.resize(std::max(floatTexelCount, static_cast<size_t>(4)));
	uintTexs.resize(std::max(uintTexelCount, static_cast<size_t>(4)));

	if (floatTexelCount > 0)
	{
		size_t texelOffset = 0;
		for (size_t i = 0; i < textures.size(); i++)
		{
			const auto &tex = textures.at(i);

			if (tex.type != TextureData::FLOAT4)
				continue;

			assert((texelOffset + static_cast<size_t>(tex.texelCount)) < floatTexs.size());
			m_TexDescriptors[i].texAddr = static_cast<uint>(texelOffset);

			memcpy(&floatTexs[texelOffset], tex.data, tex.texelCount * 4 * sizeof(float));
			texelOffset += tex.texelCount;
		}
	}

	if (uintTexelCount > 0)
	{
		size_t texelOffset = 0;
		for (size_t i = 0; i < textures.size(); i++)
		{
			const auto &tex = textures.at(i);

			if (tex.type != TextureData::UINT)
				continue;

			assert((texelOffset + static_cast<size_t>(tex.texelCount)) <= uintTexs.size());
			m_TexDescriptors[i].texAddr = static_cast<uint>(texelOffset);

			memcpy(&uintTexs[texelOffset], tex.data, tex.texelCount * sizeof(uint));
			texelOffset += tex.texelCount;
		}
	}

	m_FloatTextures = new cl::CLBuffer<float4>(m_Context, floatTexs);
	m_UintTextures = new cl::CLBuffer<uint>(m_Context, uintTexs);

	setTextures();
}

void RTContext::setMesh(size_t index, const rfw::Mesh &mesh) {}

void RTContext::setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) {}

void RTContext::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	if (!m_Skybox || (m_Skybox->size() != (width * height)))
	{
		delete m_Skybox;
		m_Skybox = new CLBuffer<glm::vec4>(m_Context, width * height);
	}

	for (int i = 0, s = int(width * height); i < s; i++)
		(*m_Skybox)[i] = vec4(pixels[i], 1.0f);

	m_Skybox->copy_to_device(false);
	setSkybox();
}

void RTContext::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
						  const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
						  const rfw::DeviceDirectionalLight *directionalLights)
{
	using namespace rfw::utils;

	m_LightCount = lightCount;

	if (!m_AreaLights || m_AreaLights->size() < lightCount.areaLightCount)
	{
		delete m_AreaLights;
		m_AreaLights = new CLBuffer<DeviceAreaLight>(m_Context, lightCount.areaLightCount);
	}

	if (!m_PointLights || m_PointLights->size() < lightCount.pointLightCount)
	{
		delete m_PointLights;
		m_PointLights = new CLBuffer<DevicePointLight>(m_Context, lightCount.pointLightCount);
	}

	if (!m_SpotLights || m_SpotLights->size() < lightCount.spotLightCount)
	{
		delete m_SpotLights;
		m_SpotLights = new CLBuffer<DeviceSpotLight>(m_Context, lightCount.spotLightCount);
	}

	if (!m_DirLights || m_DirLights->size() < lightCount.directionalLightCount)
	{
		delete m_DirLights;
		m_DirLights = new CLBuffer<DeviceDirectionalLight>(m_Context, lightCount.directionalLightCount);
	}

	if (lightCount.areaLightCount > 0)
		m_AreaLights->write(ArrayProxy<DeviceAreaLight>(lightCount.areaLightCount, areaLights));

	if (lightCount.pointLightCount > 0)
		m_PointLights->write(ArrayProxy<DevicePointLight>(lightCount.pointLightCount, pointLights));

	if (lightCount.spotLightCount > 0)
		m_SpotLights->write(ArrayProxy<DeviceSpotLight>(lightCount.spotLightCount, spotLights));

	if (lightCount.directionalLightCount > 0)
		m_DirLights->write(ArrayProxy<DeviceDirectionalLight>(lightCount.directionalLightCount, directionalLights));

	setLights();
}

void RTContext::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings RTContext::getAvailableSettings() const { return {}; }

void RTContext::setSetting(const rfw::RenderSetting &setting) {}

void RTContext::update() {}

void RTContext::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats RTContext::getStats() const { return rfw::RenderStats(); }

void RTContext::setPhase(uint phase) { m_IntersectKernel->set_argument(7, phase); }

void RTContext::setAccumulator()
{
	m_IntersectKernel->set_buffer(8, m_Accumulator);
	m_ShadeKernel->set_buffer(0, m_Accumulator);
	m_FinalizeKernel->set_buffer(1, m_Accumulator);
}

void RTContext::setSkybox()
{
	m_ShadeKernel->set_buffer(9, m_Skybox);
	m_ShadeKernel->set_argument(10, m_SkyboxWidth);
	m_ShadeKernel->set_argument(11, m_SkyboxHeight);
}

void RTContext::setLights()
{
	uvec4 lightCount = uvec4(m_LightCount.areaLightCount, m_LightCount.pointLightCount, m_LightCount.spotLightCount,
							 m_LightCount.directionalLightCount);

	m_ShadeKernel->set_argument(16, lightCount);
	m_ShadeKernel->set_buffer(17, m_AreaLights);
	m_ShadeKernel->set_buffer(18, m_PointLights);
	m_ShadeKernel->set_buffer(19, m_SpotLights);
	m_ShadeKernel->set_buffer(20, m_DirLights);
}

void RTContext::setBlueNoise()
{
	m_RayGenKernel->set_buffer(2, m_BlueNoise);
	m_ShadeKernel->set_buffer(4, m_BlueNoise);
}

void RTContext::setCamera() { m_RayGenKernel->set_buffer(1, m_Camera); }

void RTContext::setPCs()
{
	m_IntersectKernel->set_buffer(3, m_PCs);
	m_ShadeKernel->set_buffer(5, m_PCs);
}

void RTContext::setOrigins()
{
	m_RayGenKernel->set_buffer(3, m_Origins);
	m_IntersectKernel->set_buffer(5, m_Origins);
	m_ShadeKernel->set_buffer(7, m_Origins);
}

void RTContext::setDirections()
{
	m_RayGenKernel->set_buffer(4, m_Directions);
	m_IntersectKernel->set_buffer(6, m_Directions);
	m_ShadeKernel->set_buffer(8, m_Directions);
}

void RTContext::setStates()
{
	m_IntersectKernel->set_buffer(4, m_States);
	m_ShadeKernel->set_buffer(6, m_States);
}

void RTContext::setCounters() { m_ShadeKernel->set_buffer(15, m_Counters); }

void RTContext::setMaterials() { m_ShadeKernel->set_buffer(12, m_Materials); }

void RTContext::setTextures()
{
	m_ShadeKernel->set_buffer(13, m_UintTextures);
	m_ShadeKernel->set_buffer(14, m_FloatTextures);
}

void RTContext::setStride()
{
	m_IntersectKernel->set_argument(2, uint(m_Width * m_Height));
	m_ShadeKernel->set_argument(3, uint(m_Width * m_Height));
}

void RTContext::setPathLength(uint length)
{
	m_ShadeKernel->set_argument(1, length);
	m_IntersectKernel->set_argument(0, length);
}

void RTContext::setPathCount(uint pathCount)
{
	m_RayGenKernel->set_argument(0, pathCount);
	m_IntersectKernel->set_argument(1, pathCount);
	m_ShadeKernel->set_argument(2, pathCount);
}

void RTContext::resize_buffers()
{
	const uint new_count = m_Width * m_Height;
	setStride();

	m_FinalizeKernel->set_argument(2, m_Width);
	m_FinalizeKernel->set_argument(3, m_Height);

	if (m_PCs && m_PCs->size() >= new_count)
		return;

	delete m_PCs;
	delete m_Origins;
	delete m_Directions;
	delete m_States;
	delete m_Accumulator;

	m_PCs = new cl::CLBuffer<CLPotentialContribution>(m_Context, new_count);
	m_Origins = new cl::CLBuffer<float4>(m_Context, new_count);
	m_Directions = new cl::CLBuffer<float4>(m_Context, new_count);
	m_States = new cl::CLBuffer<float4>(m_Context, new_count);
	m_Accumulator = new cl::CLBuffer<float4>(m_Context, new_count);

	setAccumulator();
	setPCs();
	setOrigins();
	setDirections();
	setStates();
}
