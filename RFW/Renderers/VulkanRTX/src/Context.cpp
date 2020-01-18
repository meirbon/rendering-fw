#include "Context.h"

#include "Bindings.h"

#ifdef NDEBUG
constexpr std::array<const char *, 0> VALIDATION_LAYERS = {};
#else
constexpr std::array<const char *, 1> VALIDATION_LAYERS = {"VK_LAYER_LUNARG_standard_validation"};
#endif
const std::vector<const char *> DEVICE_EXTENSIONS = {VK_NV_RAY_TRACING_EXTENSION_NAME, VK_KHR_GET_MEMORY_REQUIREMENTS_2_EXTENSION_NAME,
													 VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME};

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType,
													const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
	const char *severity = 0, *type = 0;
	switch (messageSeverity)
	{
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT):
		return VK_FALSE;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT):
		severity = "INFO";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT):
		severity = "WARNING";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT):
		severity = "ERROR";
		break;
	default:
		break;
	}

	switch (messageType)
	{
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT):
		type = "GENERAL";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT):
		type = "VALIDATION";
		break;
	case (VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT):
		type = "PERFORMANCE";
		break;
	default:
		break;
	}

#ifndef NDEBUG
//	rfw::utils::logger::log("Vulkan Validation Layer: [Severity: %s] [Type: %s] : %s\n", severity, type, pCallbackData->pMessage);
#endif
	return VK_FALSE;
}

vkrtx::VkCamera::VkCamera(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height)
{
	pos_lensSize = glm::vec4(view.pos, view.aperture);
	right_spreadAngle = vec4(view.p2 - view.p1, view.spreadAngle);
	up = vec4(view.p3 - view.p1, 1.0f);
	p1 = vec4(view.p1, 1.0f);

	samplesTaken = samples;
	geometryEpsilon = 1e-5f;
	scrwidth = width;
	scrheight = height;
}

vkrtx::FinalizeParams::FinalizeParams(const int w, const int h, int samplespp, const float brightness, const float contrast)
{
	this->scrwidth = w;
	this->scrheight = h;

	this->spp = samplespp;
	this->pixelValueScale = 1.0f / float(this->spp);

	this->brightness = brightness;
	this->contrastFactor = (259.0f * (contrast * 256.0f + 255.0f)) / (255.0f * (259.0f - 256.0f * contrast));
}

vkrtx::Context::Context()
{
	Shader::BaseFolder = R"(vkrtxshaders/)";

	m_Initialized = false;
	createInstance();
}

vkrtx::Context::~Context() { cleanup(); }

void vkrtx::Context::init(std::shared_ptr<rfw::utils::Window> &window) { RenderContext::init(window); }

void vkrtx::Context::init(GLuint *glTextureID, uint width, uint height)
{
	m_SamplesPP = 1;
	m_SamplesTaken = 0;
	m_ScrWidth = width;
	m_ScrHeight = height;

	if (!m_Initialized)
	{
		glewInit();
		createDevice();
		initRenderer();
		m_Initialized = true;
	}
	glFlush();
	glFinish();
	m_Device.waitIdle();
	if (!m_InteropTexture)
	{
		// Create a bigger buffer than needed to prevent reallocating often
		m_InteropTexture = new InteropTexture(m_Device, *glTextureID, m_ScrWidth, m_ScrHeight);
		auto cmdBuffer = m_Device.createOneTimeCmdBuffer();
		auto queue = m_Device.getGraphicsQueue();
		m_InteropTexture->transitionImageToInitialState(cmdBuffer, queue);
		cmdBuffer.submit(queue, true);
		resizeBuffers();			// resize path trace storage buffer
		initializeDescriptorSets(); // Update descriptor sets with new target
	}
	else
	{
		m_InteropTexture->resize(*glTextureID, m_ScrWidth, m_ScrHeight);
		resizeBuffers(); // resize path trace storage buffer
	}

	finalizeDescriptorSet->bind(fOUTPUT, {m_InteropTexture->getDescriptorImageInfo()});
	CheckGL();
}

void vkrtx::Context::cleanup()
{
	glFlush(), glFinish();

	m_Device->waitIdle();

	// delete m_CounterTransferBuffer;
	// m_CounterTransferBuffer = nullptr;

	delete rtPipeline;
	rtPipeline = nullptr;
	delete rtDescriptorSet;
	rtDescriptorSet = nullptr;
	delete shadePipeline;
	shadePipeline = nullptr;
	delete shadeDescriptorSet;
	shadeDescriptorSet = nullptr;
	delete finalizePipeline;
	finalizePipeline = nullptr;
	delete finalizeDescriptorSet;
	finalizeDescriptorSet = nullptr;
	if (m_BlitCommandBuffer)
		m_Device.freeCommandBuffer(m_BlitCommandBuffer);
	m_BlitCommandBuffer = nullptr;
	delete m_TopLevelAS;
	m_TopLevelAS = nullptr;
	for (auto *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();
	delete m_OffscreenImage;
	m_OffscreenImage = nullptr;
	delete m_InvTransformsBuffer;
	m_InvTransformsBuffer = nullptr;
	delete m_AreaLightBuffer;
	m_AreaLightBuffer = nullptr;
	delete m_PointLightBuffer;
	m_PointLightBuffer = nullptr;
	delete m_SpotLightBuffer;
	m_SpotLightBuffer = nullptr;
	delete m_DirectionalLightBuffer;
	m_DirectionalLightBuffer = nullptr;
	delete m_BlueNoiseBuffer;
	m_BlueNoiseBuffer = nullptr;
	delete m_CombinedStateBuffer[0];
	m_CombinedStateBuffer[0] = nullptr;
	delete m_CombinedStateBuffer[1];
	m_CombinedStateBuffer[1] = nullptr;
	delete m_ScratchBuffer;
	m_ScratchBuffer = nullptr;
	delete m_Counters;
	m_Counters = nullptr;
	delete m_SkyboxImage;
	m_SkyboxImage = nullptr;
	delete m_UniformCamera;
	m_UniformCamera = nullptr;
	delete m_UniformFinalizeParams;
	m_UniformFinalizeParams = nullptr;
	delete m_AccumulationBuffer;
	m_AccumulationBuffer = nullptr;
	delete m_PotentialContributionBuffer;
	m_PotentialContributionBuffer = nullptr;
	delete m_Materials;
	m_Materials = nullptr;
	delete m_InteropTexture;
	m_InteropTexture = nullptr;
	if (m_VkDebugMessenger)
		m_VkInstance.destroyDebugUtilsMessengerEXT(m_VkDebugMessenger, nullptr, m_Device.getLoader());
	m_VkDebugMessenger = nullptr;

	// Vulkan device & Vulkan instance automatically get freed when this class gets destroyed
}

void vkrtx::Context::renderFrame(const rfw::Camera &cam, rfw::RenderStatus status)
{
	// Ensure OpenGL finished
	glFinish();

	using namespace rfw;
	const auto view = cam.getView();

	auto &camera = m_UniformCamera->getData()[0];
	Counters &c = m_HostCounters;

	auto queue = m_Device.getGraphicsQueue();
	if (status == Reset || m_FirstConvergingFrame)
	{
		m_SamplesTaken = 0;
	}

	const bool record = rtDescriptorSet->isDirty() || shadeDescriptorSet->isDirty() || finalizeDescriptorSet->isDirty() ||
						m_First; // Before we render we potentially have to update our command buffers
	if (record)
	{
		rtDescriptorSet->updateSetContents();
		shadeDescriptorSet->updateSetContents();
		finalizeDescriptorSet->updateSetContents();
		recordCommandBuffers();
		m_First = false;
	}

	// Get queue and command buffer for this frame
	OneTimeCommandBuffer cmdBuffer = m_Device.createOneTimeCmdBuffer(vk::CommandBufferLevel::ePrimary, VulkanDevice::COMPUTE);

	uint pathCount = m_ScrWidth * m_ScrHeight * 1;
	uint32_t pushConstant[3];

	utils::Timer t;
	m_Stats.clear();

	for (uint i = 0; i < m_SamplesPP; i++)
	{
		// Initialize camera
		camera = VkCamera(view, m_SamplesTaken, m_GeometryEpsilon, m_ScrWidth, m_ScrHeight); // reset camera
		m_UniformCamera->copyToDevice();

		// Initialize counters
		c.Reset(m_LightCounts, m_ScrWidth, m_ScrHeight, 10.0f);
		c.probePixelIdx = m_ProbePos.x + m_ProbePos.y * m_ScrWidth;
		m_Counters->copyToDevice(&c);

		if (i != 0)
			cmdBuffer.begin();

		// Primary ray stage
		if (m_SamplesTaken <= 1)
			cmdBuffer->fillBuffer(*m_AccumulationBuffer, 0, m_ScrWidth * m_ScrHeight * sizeof(glm::vec4), 0);
		pushConstant[0] = c.pathLength;
		pushConstant[1] = pathCount;
		pushConstant[2] = STAGE_PRIMARY_RAY;
		rtPipeline->recordPushConstant(cmdBuffer, 0, 3 * sizeof(uint32_t),
									   pushConstant); // Push intersection stage to shader
		rtPipeline->recordTraceCommand(cmdBuffer.getVkCommandBuffer(), pathCount + (64 - (pathCount % 64)));

		// submit primary rays to queue
		t.reset();
		cmdBuffer.submit(queue, true);
		m_Stats.primaryTime += t.elapsed();
		m_Stats.primaryCount += pathCount;

		// Record shade stage
		cmdBuffer.begin();
		shadePipeline->recordPushConstant(cmdBuffer, 0, 2 * sizeof(uint32_t), pushConstant);
		shadePipeline->recordDispatchCommand(cmdBuffer, pathCount + (64 - (pathCount % 64)));

		// Make sure shading finished before copying counters
		cmdBuffer->pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, {}, {}, {});

		// submit command buffer
		t.reset();
		cmdBuffer.submit(queue, true);
		m_Stats.shadeTime += t.elapsed();

		m_Counters->copyToHost(&c);

		// Prepare extension rays
		pathCount = c.extensionRays;
		c.extensionRays = 0;
		c.pathLength++;

		if (m_SamplesTaken == 0)
		{
			m_ProbedInstance = c.probedInstid;
			m_ProbedTriangle = c.probedTriid;
			m_ProbedDist = c.probedDist;
		}

		for (uint j = 1; j <= MAXPATHLENGTH; j++)
		{
			if (pathCount > 1)
			{
				m_Counters->copyToDevice(&c);

				// Extension ray stage
				cmdBuffer.begin();
				pushConstant[0] = c.pathLength;
				pushConstant[1] = pathCount;
				pushConstant[2] = STAGE_SECONDARY_RAY;
				rtPipeline->recordPushConstant(cmdBuffer.getVkCommandBuffer(), 0, 3 * sizeof(uint32_t),
											   pushConstant); // Push intersection stage to shader
				rtPipeline->recordTraceCommand(cmdBuffer.getVkCommandBuffer(), pathCount + (64 - (pathCount % 64)));

				// Run command buffer
				t.reset();
				cmdBuffer.submit(queue, true);
				if (j == 1)
				{
					m_Stats.secondaryTime += t.elapsed();
					m_Stats.secondaryCount += pathCount;
				}
				else
				{
					m_Stats.deepTime += t.elapsed();
					m_Stats.deepCount += pathCount;
				}

				cmdBuffer.begin();
				// Shade extension rays
				shadePipeline->recordPushConstant(cmdBuffer, 0, 2 * sizeof(uint32_t), pushConstant);
				shadePipeline->recordDispatchCommand(cmdBuffer, pathCount + (64 - (pathCount % 64)));

				// Submit shade command buffer
				t.reset();
				cmdBuffer.submit(queue, true); // Run command buffer
				m_Stats.shadeTime += t.elapsed();

				m_Counters->copyToHost(&c);

				pathCount = c.extensionRays; // Get number of extension rays generated
				c.pathCount = pathCount;
				c.extensionRays = 0; // reset extension counter
				c.pathLength++;		 // Increment path length
			}
			else
			{
				break; // All paths were terminated
			}
		}

		// Prepare shadow rays
		pathCount = c.shadowRays; // Get number of shadow rays generated
		if (pathCount > 0)
		{
			cmdBuffer.begin();
			pushConstant[1] = pathCount;
			pushConstant[2] = STAGE_SHADOW_RAY;
			rtPipeline->recordPushConstant(cmdBuffer.getVkCommandBuffer(), 0, 3 * sizeof(uint32_t),
										   pushConstant); // Push intersection stage to shader
			rtPipeline->recordTraceCommand(cmdBuffer.getVkCommandBuffer(), pathCount + (64 - (pathCount % 64)));

			// submit shadow rays
			t.reset();
			cmdBuffer.submit(queue, true); // Run command buffer
			m_Stats.shadowTime += t.elapsed();
			m_Stats.shadowCount += pathCount;
		}

		m_SamplesTaken++;
	}

	// Initialize params for finalize stage
	m_UniformFinalizeParams->getData()[0] = FinalizeParams(m_ScrWidth, m_ScrHeight, m_SamplesTaken, cam.brightness, cam.contrast);
	m_UniformFinalizeParams->copyToDevice();

	t.reset();
	m_Device.submitCommandBuffer(m_BlitCommandBuffer, queue);
	queue.waitIdle();
	m_Stats.finalizeTime = t.elapsed();
	cmdBuffer.begin();
}

void vkrtx::Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
	delete m_Materials;
	std::vector<rfw::DeviceMaterial> materialData(materials.size());
	materialData.resize(materials.size());
	memcpy(materialData.data(), materials.data(), materials.size() * sizeof(rfw::DeviceMaterial));

	const std::vector<rfw::TextureData> &texDescs = m_TexDescriptors;

	for (int i = 0; i < materials.size(); i++)
	{
		auto &mat = reinterpret_cast<rfw::Material &>(materialData.at(i));
		const rfw::MaterialTexIds &ids = texDescriptors[i];
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

	m_Materials =
		new VmaBuffer<rfw::DeviceMaterial>(m_Device, materialData.size(), vk::MemoryPropertyFlagBits::eDeviceLocal,
										   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_Materials->copyToDevice(materialData.data(), materialData.size() * sizeof(rfw::DeviceMaterial));

	shadeDescriptorSet->bind(cMATERIALS, {m_Materials->getDescriptorBufferInfo()});

	std::vector<rfw::Material> mats(materials.size());
	m_Materials->copyToHost(mats.data());
}

void vkrtx::Context::setTextures(const std::vector<rfw::TextureData> &textures)
{
	m_TexDescriptors = textures;

	delete m_RGBA32Buffer;
	delete m_RGBA128Buffer;
	m_RGBA32Buffer = nullptr;
	m_RGBA128Buffer = nullptr;

	size_t uintTexelCount = 0;
	size_t floatTexelCount = 0;

	std::vector<glm::vec4> floatTexs;
	std::vector<uint> uintTexs;

	for (const auto &tex : textures)
	{
		switch (tex.type)
		{
		case (rfw::TextureData::FLOAT4):
			floatTexelCount += tex.texelCount;
			break;
		case (rfw::TextureData::UINT):
			uintTexelCount += tex.texelCount;
			break;
		}
	}

	floatTexs.resize(glm::max(floatTexelCount, size_t(4)));
	uintTexs.resize(glm::max(uintTexelCount, size_t(4)));

	if (floatTexelCount > 0)
	{
		size_t texelOffset = 0;
		for (size_t i = 0; i < textures.size(); i++)
		{
			const auto &tex = textures.at(i);

			if (tex.type != rfw::TextureData::FLOAT4)
				continue;

			assert((texelOffset + static_cast<size_t>(tex.texelCount)) < floatTexs.size());
			m_TexDescriptors.at(i).texAddr = static_cast<uint>(texelOffset);

			memcpy(&floatTexs.at(texelOffset), tex.data, tex.texelCount * 4 * sizeof(float));
			texelOffset += tex.texelCount;
		}
	}

	if (uintTexelCount > 0)
	{
		size_t texelOffset = 0;
		for (size_t i = 0; i < textures.size(); i++)
		{
			const auto &tex = textures.at(i);

			if (tex.type != rfw::TextureData::UINT)
				continue;

			assert((texelOffset + static_cast<size_t>(tex.texelCount)) <= uintTexs.size());
			m_TexDescriptors.at(i).texAddr = static_cast<uint>(texelOffset);

			memcpy(&uintTexs.at(texelOffset), tex.data, tex.texelCount * sizeof(uint));
			texelOffset += tex.texelCount;
		}
	}

	uintTexelCount = glm::max(uintTexelCount, size_t(1));
	floatTexelCount = glm::max(floatTexelCount, size_t(1));

	m_RGBA32Buffer = new VmaBuffer<uint>(m_Device, uintTexelCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
										 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_ONLY);
	m_RGBA128Buffer = new VmaBuffer<glm::vec4>(m_Device, floatTexelCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
											   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_ONLY);

	if (!uintTexs.empty())
		m_RGBA32Buffer->copyToDevice(uintTexs.data(), m_RGBA32Buffer->getSize());
	if (!floatTexs.empty())
		m_RGBA128Buffer->copyToDevice(floatTexs.data(), m_RGBA128Buffer->getSize());

	shadeDescriptorSet->bind(cTEXTURE_RGBA32, {m_RGBA32Buffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cTEXTURE_RGBA128, {m_RGBA128Buffer->getDescriptorBufferInfo()});
}

void vkrtx::Context::setMesh(size_t index, const rfw::Mesh &mesh)
{
	if (index >= m_Meshes.size())
	{
		while (index >= m_Meshes.size())
		{
			m_Meshes.push_back(new Mesh(m_Device));
			m_MeshChanged.push_back(false);
		}
	}

	m_Meshes[index]->setGeometry(mesh, *m_ScratchBuffer);
	m_MeshChanged[index] = true;
}

void vkrtx::Context::setInstance(size_t index, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
{
	if (index >= m_Instances.size())
	{
		m_Instances.emplace_back();
		m_InstanceMeshIndices.emplace_back();
		m_InvTransforms.emplace_back(1.0f);

		if (m_InvTransformsBuffer->getElementCount() < m_Instances.size())
		{
			delete m_InvTransformsBuffer;
			m_InvTransformsBuffer = new VmaBuffer<mat4>(
				m_Device, m_Instances.size() + (m_Instances.size() % 32), vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
				vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);
		}
	}

	m_InstanceMeshIndices.at(index) = meshIdx;
	auto &curInstance = m_Instances.at(index);

	curInstance.instanceId = static_cast<uint32_t>(index);
	curInstance.mask = 0xFF;
	curInstance.instanceOffset = 0;
	curInstance.flags = static_cast<uint32_t>(vk::GeometryInstanceFlagBitsNV::eTriangleCullDisable);

	// Update matrix
	const auto tmpTransform = transpose(transform);
	memcpy(curInstance.transform, value_ptr(tmpTransform), sizeof(curInstance.transform));
	m_InvTransforms[index] = mat4(inverse_transform);

	// Update acceleration structure handle
	curInstance.accelerationStructureHandle = m_Meshes.at(meshIdx)->accelerationStructure->getHandle();
}

void vkrtx::Context::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	std::vector<glm::vec4> data(size_t(width * height));
	for (uint i = 0; i < (width * height); i++)
		data[i] = glm::vec4(pixels[i].x, pixels[i].y, pixels[i].z, 0.0f);

	delete m_SkyboxImage;
	// Create a Vulkan image that can be sampled
	m_SkyboxImage =
		new Image(m_Device, vk::ImageType::e2D, vk::Format::eR32G32B32A32Sfloat, vk::Extent3D(static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1),
				  vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst, vk::MemoryPropertyFlagBits::eDeviceLocal,
				  Image::SKYDOME);

	vk::ImageSubresourceRange range{};
	range.aspectMask = vk::ImageAspectFlagBits::eColor;
	range.baseMipLevel = 0;
	range.levelCount = 1;
	range.baseArrayLayer = 0;
	range.layerCount = 1;

	// Set image data
	m_SkyboxImage->setData(data, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
	// Create an image view that can be sampled
	m_SkyboxImage->createImageView(vk::ImageViewType::e2D, vk::Format::eR32G32B32A32Sfloat, range);
	// Create sampler to be used in shader
	m_SkyboxImage->createSampler(vk::Filter::eLinear, vk::Filter::eNearest, vk::SamplerMipmapMode::eLinear, vk::SamplerAddressMode::eClampToEdge);
	// Make sure image is usable by shader
	m_SkyboxImage->transitionToLayout(vk::ImageLayout::eShaderReadOnlyOptimal, vk::AccessFlags());
	// Update descriptor set
	shadeDescriptorSet->bind(cSKYBOX, {m_SkyboxImage->getDescriptorImageInfo()});
}

void vkrtx::Context::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
							   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights)
{
	m_LightCounts = lightCount;

	if (m_AreaLightBuffer->getElementCount() < lightCount.areaLightCount)
	{
		delete m_AreaLightBuffer;
		m_AreaLightBuffer = new VmaBuffer<rfw::DeviceAreaLight>(m_Device, m_LightCounts.areaLightCount,
																vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
																vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}
	if (m_PointLightBuffer->getElementCount() < lightCount.pointLightCount)
	{
		delete m_PointLightBuffer;
		m_PointLightBuffer = new VmaBuffer<rfw::DevicePointLight>(m_Device, m_LightCounts.pointLightCount,
																  vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
																  vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}
	if (m_SpotLightBuffer->getElementCount() < lightCount.spotLightCount)
	{
		delete m_SpotLightBuffer;
		m_SpotLightBuffer = new VmaBuffer<rfw::DeviceSpotLight>(m_Device, lightCount.spotLightCount,
																vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
																vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}
	if (m_DirectionalLightBuffer->getElementCount() < lightCount.directionalLightCount)
	{
		delete m_DirectionalLightBuffer;
		m_DirectionalLightBuffer = new VmaBuffer<rfw::DeviceDirectionalLight>(
			m_Device, lightCount.directionalLightCount, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
			vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	}

	// Copy to device in case lights exist
	if (lightCount.areaLightCount > 0)
		m_AreaLightBuffer->copyToDevice(areaLights, m_AreaLightBuffer->getSize());
	if (lightCount.pointLightCount > 0)
		m_PointLightBuffer->copyToDevice(pointLights, m_PointLightBuffer->getSize());
	if (lightCount.spotLightCount > 0)
		m_SpotLightBuffer->copyToDevice(spotLights, m_SpotLightBuffer->getSize());
	if (lightCount.directionalLightCount > 0)
		m_DirectionalLightBuffer->copyToDevice(directionalLights, m_DirectionalLightBuffer->getSize());

	// Update descriptor set
	shadeDescriptorSet->bind(cAREALIGHT_BUFFER, {m_AreaLightBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cPOINTLIGHT_BUFFER, {m_PointLightBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cSPOTLIGHT_BUFFER, {m_SpotLightBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cDIRECTIONALLIGHT_BUFFER, {m_DirectionalLightBuffer->getDescriptorBufferInfo()});
}

void vkrtx::Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const
{
	(*instanceIndex) = m_ProbedInstance;
	(*primitiveIndex) = m_ProbedTriangle;
	(*distance) = m_ProbedDist;
}

rfw::AvailableRenderSettings vkrtx::Context::getAvailableSettings() const { return rfw::AvailableRenderSettings(); }

void vkrtx::Context::setSetting(const rfw::RenderSetting &setting) {}

void vkrtx::Context::update()
{
	for (uint i = 0; i < m_Instances.size(); i++)
	{
		// Meshes might have changed in the mean time
		const auto meshIdx = m_InstanceMeshIndices.at(i);
		if (!m_MeshChanged.at(meshIdx))
			continue;

		auto &instance = m_Instances.at(i);
		auto *mesh = m_Meshes.at(meshIdx);

		// Update acceleration structure handle
		instance.accelerationStructureHandle = mesh->accelerationStructure->getHandle();
		assert(instance.accelerationStructureHandle);
	}

	bool triangleBuffersDirty = false;						// Initially we presume triangle buffers are up to date
	if (m_TriangleBufferInfos.size() != m_Instances.size()) // Update every triangle buffer info
	{
		triangleBuffersDirty = true;
		m_TriangleBufferInfos.resize(m_Instances.size());
		for (uint i = 0; i < m_Instances.size(); i++)
		{
			const auto *mesh = m_Meshes.at(m_InstanceMeshIndices.at(i));
			m_TriangleBufferInfos.at(i) = mesh->triangles.getDescriptorBufferInfo();
		}
	}
	else
	{
		for (uint i = 0; i < m_Instances.size(); i++)
		{
			const auto meshIdx = m_InstanceMeshIndices.at(i);
			if (m_MeshChanged.at(meshIdx))
			{
				const auto *mesh = m_Meshes.at(m_InstanceMeshIndices.at(i));
				m_TriangleBufferInfos.at(i) = mesh->triangles.getDescriptorBufferInfo();
				triangleBuffersDirty = true; // Set triangle buffer write flag
			}
		}
	}

	m_MeshChanged.clear();
	m_MeshChanged.resize(m_Meshes.size(), false);

	// Update triangle buffers
	if (triangleBuffersDirty)
		shadeDescriptorSet->bind(cTRIANGLES, m_TriangleBufferInfos);

	assert(m_InvTransformsBuffer->getElementCount() >= m_InvTransforms.size());

	m_InvTransformsBuffer->copyToDevice(m_InvTransforms.data(),
										m_InvTransforms.size() * sizeof(mat4)); // Update inverse transforms
	shadeDescriptorSet->bind(cINVERSE_TRANSFORMS, {m_InvTransformsBuffer->getDescriptorBufferInfo()});

	if (m_TopLevelAS->getInstanceCount() != m_Instances.size()) // Recreate top level AS in case our number of instances changed
	{
		delete m_TopLevelAS;
		m_TopLevelAS = new TopLevelAS(m_Device, FastTrace, static_cast<uint32_t>(m_Instances.size()));
		m_TopLevelAS->updateInstances(m_Instances);
		m_TopLevelAS->build(*m_ScratchBuffer);

		rtDescriptorSet->bind(rtACCELERATION_STRUCTURE, {m_TopLevelAS->getDescriptorBufferInfo()});
	}
	else if (!m_TopLevelAS->canUpdate()) // Recreate top level AS in case it cannot be updated
	{
		delete m_TopLevelAS;
		m_TopLevelAS = new TopLevelAS(m_Device, FastTrace, static_cast<uint32_t>(m_Instances.size()));
		m_TopLevelAS->updateInstances(m_Instances);
		m_TopLevelAS->build(*m_ScratchBuffer);

		// Update descriptor set
		rtDescriptorSet->bind(rtACCELERATION_STRUCTURE, {m_TopLevelAS->getDescriptorBufferInfo()});
	}
	else // rebuild (refit) our top level AS
	{
		m_TopLevelAS->updateInstances(m_Instances);
		assert(m_TopLevelAS->canUpdate());
		m_TopLevelAS->rebuild(*m_ScratchBuffer);

		// No descriptor write needed, same acceleration structure object
	}
}

void vkrtx::Context::setProbePos(glm::uvec2 probePos) { m_ProbePos = probePos; }

rfw::RenderStats vkrtx::Context::getStats() const { return m_Stats; }

void vkrtx::Context::initRenderer()
{
#ifndef NDEBUG
	createDebugReportCallback();
#endif
	createCommandBuffers();								   // Initialize blit buffers
	m_TopLevelAS = new TopLevelAS(m_Device, FastestTrace); // Create a top level AS, Vulkan doesn't like unbound buffers
	createDescriptorSets();								   // Create bindings for shaders
	createBuffers();									   // Create uniforms like our camera
	createRayTracingPipeline();							   // Create ray intersection pipeline
	createShadePipeline();								   // Create compute pipeline; wavefront shading
	createFinalizePipeline();							   // Create compute pipeline; plot accumulation buffer to image
	m_TopLevelAS->build(*m_ScratchBuffer);				   // build top level AS

	// Set initial sky box, Vulkan does not like having unbound buffers
	auto dummy = glm::vec3(0.0f);
	setSkyDome({dummy}, 1, 1);

	m_Initialized = true;
}

void vkrtx::Context::createInstance()
{
	std::vector<const char *> extensions;

#ifndef NDEBUG
	extensions.push_back(VK_EXT_DEBUG_REPORT_EXTENSION_NAME);
	extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
#endif

	extensions.push_back(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
	extensions.push_back(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);

	auto appInfo = vk::ApplicationInfo("Vulkan RTX", 0, "No Engine", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_1);
	auto createInfo = vk::InstanceCreateInfo();
	createInfo.setPApplicationInfo(&appInfo);

	// Configure required extensions;
	createInfo.setEnabledExtensionCount(static_cast<uint32_t>(extensions.size()));
	createInfo.setPpEnabledExtensionNames(extensions.data());

#ifndef NDEBUG
	setupValidationLayers(createInfo); // Configure Vulkan validation layers
#endif

	m_VkInstance = vk::createInstance(createInfo);
	if (!m_VkInstance)
		FAILURE("Could not initialize Vulkan.");
	printf("Successfully created Vulkan instance.\n");
}

void vkrtx::Context::setupValidationLayers(vk::InstanceCreateInfo &createInfo)
{
	createInfo.setEnabledLayerCount(0);

	// Check if requested validation layers are present
#ifndef NDEBUG
	// Get supported layers
	const auto availableLayers = vk::enumerateInstanceLayerProperties();

	const auto hasLayer = [&availableLayers](const char *layerName) -> bool {
		for (const auto &layer : availableLayers)
		{
			if (strcmp(layerName, layer.layerName) == 0)
				return true;
		}

		return false;
	};

	bool layersFound = true;
	for (auto layer : VALIDATION_LAYERS)
	{
		if (!hasLayer(layer))
		{
			createInfo.setEnabledLayerCount(0), layersFound = false;
			printf("Could not enable validation layer: \"%s\"\n", layer);
			break;
		}
	}

	if (layersFound)
	{
		// All layers available
		createInfo.setEnabledLayerCount(static_cast<uint32_t>(VALIDATION_LAYERS.size()));
		createInfo.setPpEnabledLayerNames(VALIDATION_LAYERS.data());
	}
#endif
}

void vkrtx::Context::createDebugReportCallback()
{
	m_VkDebugMessenger = nullptr;

#ifndef NDEBUG
	vk::DebugUtilsMessengerCreateInfoEXT dbgMessengerCreateInfo{};
	dbgMessengerCreateInfo.setMessageSeverity(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError |
											  vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning);
	dbgMessengerCreateInfo.setMessageType(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
										  vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
	dbgMessengerCreateInfo.setPfnUserCallback(debugCallback);
	dbgMessengerCreateInfo.setPUserData(nullptr);

	m_VkDebugMessenger = m_VkInstance.createDebugUtilsMessengerEXT(dbgMessengerCreateInfo, nullptr, m_Device.getLoader());
	if (!m_VkDebugMessenger)
		printf("Could not setup Vulkan debug utils messenger.\n");
#endif
}

void vkrtx::Context::createDevice()
{
	// Start with application defined required extensions
	std::vector<const char *> dev_extensions = DEVICE_EXTENSIONS;
	for (const auto ext : InteropTexture::getRequiredExtensions())
		dev_extensions.push_back(ext);

	// Retrieve a physical device that supports our requested extensions
	const auto physicalDevice = VulkanDevice::pickDeviceWithExtensions(m_VkInstance, dev_extensions);

	// Sanity check
	if (!physicalDevice.has_value())
		FAILURE("No supported Vulkan devices available.");

	// Create device
	m_Device = VulkanDevice(m_VkInstance, physicalDevice.value(), dev_extensions);
}

void vkrtx::Context::createCommandBuffers()
{
	if (m_BlitCommandBuffer)
		m_Device.freeCommandBuffer(m_BlitCommandBuffer);
	m_BlitCommandBuffer = m_Device.createCommandBuffer(vk::CommandBufferLevel::ePrimary);
}

void vkrtx::Context::resizeBuffers()
{
	const auto newPixelCount = uint32_t((m_ScrWidth * m_ScrHeight) * 1.3f); // Make buffer bigger than needed to prevent reallocating often
	const auto oldPixelCount = m_AccumulationBuffer->getSize() / sizeof(glm::vec4);

	if (oldPixelCount >= newPixelCount)
		return; // No need to resize buffers

	delete m_CombinedStateBuffer[0];
	delete m_CombinedStateBuffer[1];
	delete m_AccumulationBuffer;
	delete m_PotentialContributionBuffer;

	const auto limits = m_Device.getPhysicalDevice().getProperties().limits;

	// Create 2 path trace state buffers, these buffers are ping-ponged every path iteration
	const auto combinedAlignedSize = newPixelCount * 4 + ((newPixelCount * 4) % limits.minStorageBufferOffsetAlignment);
	m_CombinedStateBuffer[0] = new VmaBuffer<glm::vec4>(m_Device, combinedAlignedSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
														vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
	m_CombinedStateBuffer[1] = new VmaBuffer<glm::vec4>(m_Device, combinedAlignedSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
														vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);

	// Accumulation buffer for rendered image
	m_AccumulationBuffer = new VmaBuffer<glm::vec4>(m_Device, newPixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
													vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_ONLY);
	// Shadow ray buffer
	m_PotentialContributionBuffer = new VmaBuffer<PotentialContribution>(m_Device, MAXPATHLENGTH * newPixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
																		 vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);

	const auto unalignedSize = m_ScrWidth * m_ScrHeight * sizeof(float) * 4;
	const vk::DeviceSize singleSize = unalignedSize + (unalignedSize % (4 * limits.minUniformBufferOffsetAlignment));

	// Buffers got recreated so we need to update our descriptor sets
	rtDescriptorSet->bind(rtPATH_STATES,
						  {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(0, singleSize), m_CombinedStateBuffer[1]->getDescriptorBufferInfo(0, singleSize)});
	rtDescriptorSet->bind(rtPATH_ORIGINS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(singleSize, singleSize),
										   m_CombinedStateBuffer[1]->getDescriptorBufferInfo(singleSize, singleSize)});
	rtDescriptorSet->bind(rtPATH_DIRECTIONS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(2 * singleSize, singleSize),
											  m_CombinedStateBuffer[1]->getDescriptorBufferInfo(2 * singleSize, singleSize)});
	rtDescriptorSet->bind(rtACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
	rtDescriptorSet->bind(rtPOTENTIAL_CONTRIBUTIONS, {m_PotentialContributionBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cPOTENTIAL_CONTRIBUTIONS, {m_PotentialContributionBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(
		cPATH_STATES, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(0, singleSize), m_CombinedStateBuffer[1]->getDescriptorBufferInfo(0, singleSize)});
	shadeDescriptorSet->bind(cPATH_ORIGINS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(singleSize, singleSize),
											 m_CombinedStateBuffer[1]->getDescriptorBufferInfo(singleSize, singleSize)});
	shadeDescriptorSet->bind(cPATH_DIRECTIONS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(2 * singleSize, singleSize),
												m_CombinedStateBuffer[1]->getDescriptorBufferInfo(2 * singleSize, singleSize)});
	shadeDescriptorSet->bind(cPATH_THROUGHPUTS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(3 * singleSize, singleSize),
												 m_CombinedStateBuffer[1]->getDescriptorBufferInfo(3 * singleSize, singleSize)});
	finalizeDescriptorSet->bind(fACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
}

void vkrtx::Context::createRayTracingPipeline()
{
	rtPipeline = new RTXPipeline(m_Device);
	// Setup ray generation shader
	const auto rgenShader = Shader(m_Device, "rt_shaders.rgen");
	const auto chitShader = Shader(m_Device, "rt_shaders.rchit");
	const auto missShader = Shader(m_Device, "rt_shaders.rmiss");
	const auto shadowShader = Shader(m_Device, "rt_shadow.rmiss");

	// Setup pipeline
	rtPipeline->addRayGenShaderStage(rgenShader);
	const auto hitGroup = RTXHitGroup(nullptr, &chitShader);
	rtPipeline->addHitGroup(hitGroup);
	rtPipeline->addMissShaderStage(missShader);
	rtPipeline->addMissShaderStage(shadowShader);
	rtPipeline->addEmptyHitGroup();
	rtPipeline->addPushConstant(vk::PushConstantRange(vk::ShaderStageFlagBits::eRaygenNV, 0, 3 * sizeof(uint32_t)));
	rtPipeline->setMaxRecursionDepth(5u);
	rtPipeline->addDescriptorSet(rtDescriptorSet);
	rtPipeline->finalize();
}

void vkrtx::Context::createShadePipeline()
{
	auto computeShader = Shader(m_Device, "rt_shade.comp");
	shadePipeline = new ComputePipeline(m_Device, computeShader);
	shadePipeline->addDescriptorSet(shadeDescriptorSet);
	shadePipeline->addPushConstant(vk::PushConstantRange(vk::ShaderStageFlagBits::eCompute, 0, sizeof(uint32_t) * 2));
	shadePipeline->finalize();
}

void vkrtx::Context::createFinalizePipeline()
{
	auto computeShader = Shader(m_Device, "rt_finalize.comp");
	finalizePipeline = new ComputePipeline(m_Device, computeShader);
	finalizePipeline->addDescriptorSet(finalizeDescriptorSet);
	finalizePipeline->finalize();
}

void vkrtx::Context::createDescriptorSets()
{
	rtDescriptorSet = new DescriptorSet(m_Device);
	shadeDescriptorSet = new DescriptorSet(m_Device);
	finalizeDescriptorSet = new DescriptorSet(m_Device);

	rtDescriptorSet->addBinding(rtACCELERATION_STRUCTURE, 1, vk::DescriptorType::eAccelerationStructureNV,
								vk::ShaderStageFlagBits::eRaygenNV | vk::ShaderStageFlagBits::eClosestHitNV);
	rtDescriptorSet->addBinding(rtCAMERA, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtPATH_STATES, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtPATH_ORIGINS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtPATH_DIRECTIONS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtPOTENTIAL_CONTRIBUTIONS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->addBinding(rtBLUENOISE, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eRaygenNV);
	rtDescriptorSet->finalize();

	const auto limits = m_Device.getPhysicalDevice().getProperties().limits;
	assert(limits.maxDescriptorSetStorageBuffers > MAX_TRIANGLE_BUFFERS);
	shadeDescriptorSet->addBinding(cCOUNTERS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cCAMERA, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPATH_STATES, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPATH_ORIGINS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPATH_DIRECTIONS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPATH_THROUGHPUTS, 2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPOTENTIAL_CONTRIBUTIONS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cMATERIALS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cSKYBOX, 1, vk::DescriptorType::eCombinedImageSampler, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cTRIANGLES, MAX_TRIANGLE_BUFFERS, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cINVERSE_TRANSFORMS, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cTEXTURE_RGBA32, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cTEXTURE_RGBA128, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cAREALIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cPOINTLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cSPOTLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cDIRECTIONALLIGHT_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->addBinding(cBLUENOISE, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	shadeDescriptorSet->finalize();

	finalizeDescriptorSet->addBinding(fACCUMULATION_BUFFER, 1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute);
	finalizeDescriptorSet->addBinding(fUNIFORM_CONSTANTS, 1, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eCompute);
	finalizeDescriptorSet->addBinding(fOUTPUT, 1, vk::DescriptorType::eStorageImage, vk::ShaderStageFlagBits::eCompute);
	finalizeDescriptorSet->finalize();
}

void vkrtx::Context::recordCommandBuffers()
{
	vk::CommandBufferBeginInfo beginInfo{};
	// Start recording
	m_BlitCommandBuffer.begin(beginInfo);

	// Make sure off-screen render image is ready to be used
	const auto subresourceRange = vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
	m_InteropTexture->recordTransitionToVulkan(m_BlitCommandBuffer);

	m_BlitCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, vk::DependencyFlags(), 0, nullptr,
										0, nullptr, 0, nullptr);

	// Dispatch finalize image shader
	finalizePipeline->recordDispatchCommand(m_BlitCommandBuffer, m_ScrWidth, m_ScrHeight);
	m_InteropTexture->recordTransitionToGL(m_BlitCommandBuffer); // Make sure interop texture is ready to be used by Vulkan
	m_BlitCommandBuffer.end();
}

void vkrtx::Context::createBuffers()
{
	m_ScratchBuffer =
		new VmaBuffer<uint8_t>(m_Device, 65336, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eRayTracingNV, VMA_MEMORY_USAGE_GPU_ONLY);

	const auto pixelCount = static_cast<vk::DeviceSize>(m_ScrWidth * m_ScrHeight);
	m_InvTransformsBuffer = new VmaBuffer<mat4>(m_Device, 32, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
												vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_CPU_TO_GPU);

	m_UniformCamera = new UniformObject<VkCamera>(m_Device);
	m_UniformFinalizeParams = new UniformObject<FinalizeParams>(m_Device);
	m_Counters = new VmaBuffer<Counters>(
		m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
		vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_ONLY);

	// bind uniforms
	rtDescriptorSet->bind(rtCAMERA, {m_UniformCamera->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cCAMERA, {m_UniformCamera->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cCOUNTERS, {m_Counters->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(fUNIFORM_CONSTANTS, {m_UniformFinalizeParams->getDescriptorBufferInfo()});

	m_Materials = new VmaBuffer<rfw::DeviceMaterial>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
													 vk::BufferUsageFlagBits::eStorageBuffer);

	// Texture buffers
	m_RGBA32Buffer = new VmaBuffer<uint>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
										 vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);
	m_RGBA128Buffer = new VmaBuffer<glm::vec4>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal,
											   vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst);

	// Wavefront buffers
	m_AccumulationBuffer = new VmaBuffer<glm::vec4>(m_Device, pixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal, vk::BufferUsageFlagBits::eStorageBuffer,
													VMA_MEMORY_USAGE_GPU_ONLY);
	m_PotentialContributionBuffer = new VmaBuffer<PotentialContribution>(m_Device, MAXPATHLENGTH * pixelCount, vk::MemoryPropertyFlagBits::eDeviceLocal,
																		 vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);

	const auto limits = m_Device.getPhysicalDevice().getProperties().limits;
	const auto combinedAlignedSize = pixelCount * 4 + ((pixelCount * 4) % limits.minStorageBufferOffsetAlignment);
	m_CombinedStateBuffer[0] = new VmaBuffer<glm::vec4>(m_Device, combinedAlignedSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
														vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
	m_CombinedStateBuffer[1] = new VmaBuffer<glm::vec4>(m_Device, combinedAlignedSize, vk::MemoryPropertyFlagBits::eDeviceLocal,
														vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_GPU_ONLY);

	// Light buffers
	m_AreaLightBuffer = new VmaBuffer<rfw::DeviceAreaLight>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
															vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_PointLightBuffer = new VmaBuffer<rfw::DevicePointLight>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
															  vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_SpotLightBuffer = new VmaBuffer<rfw::DeviceSpotLight>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
															vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
	m_DirectionalLightBuffer =
		new VmaBuffer<rfw::DeviceDirectionalLight>(m_Device, 1, vk::MemoryPropertyFlagBits::eDeviceLocal | vk::MemoryPropertyFlagBits::eHostVisible,
												   vk::BufferUsageFlagBits::eStorageBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);

	// Blue Noise
	const auto blueNoise = createBlueNoiseBuffer();
	m_BlueNoiseBuffer = new VmaBuffer<uint>(m_Device, 65536 * 5, vk::MemoryPropertyFlagBits::eDeviceLocal,
											vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VMA_MEMORY_USAGE_GPU_ONLY);
	m_BlueNoiseBuffer->copyToDevice(blueNoise.data(), m_BlueNoiseBuffer->getSize());
}

void vkrtx::Context::initializeDescriptorSets()
{
	rtDescriptorSet->bind(rtACCELERATION_STRUCTURE, {m_TopLevelAS->getDescriptorBufferInfo()});
	rtDescriptorSet->bind(rtCAMERA, {m_UniformCamera->getDescriptorBufferInfo()});
	const auto limits = m_Device.getPhysicalDevice().getProperties().limits;
	const auto unalignedSize = m_ScrWidth * m_ScrHeight * sizeof(glm::vec4);
	const vk::DeviceSize singleSize = unalignedSize + (unalignedSize % limits.minUniformBufferOffsetAlignment);
	rtDescriptorSet->bind(rtPATH_STATES,
						  {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(0, singleSize), m_CombinedStateBuffer[1]->getDescriptorBufferInfo(0, singleSize)});
	rtDescriptorSet->bind(rtPATH_ORIGINS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(singleSize, singleSize),
										   m_CombinedStateBuffer[1]->getDescriptorBufferInfo(singleSize, singleSize)});
	rtDescriptorSet->bind(rtPATH_DIRECTIONS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(2 * singleSize, singleSize),
											  m_CombinedStateBuffer[1]->getDescriptorBufferInfo(2 * singleSize, singleSize)});
	rtDescriptorSet->bind(rtPOTENTIAL_CONTRIBUTIONS, {m_PotentialContributionBuffer->getDescriptorBufferInfo()});
	rtDescriptorSet->bind(rtACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
	rtDescriptorSet->bind(rtBLUENOISE, {m_BlueNoiseBuffer->getDescriptorBufferInfo()});

	// Update descriptor set contents
	rtDescriptorSet->updateSetContents();

	if (m_Meshes.empty())
	{
		m_Meshes.push_back(new vkrtx::Mesh(m_Device)); // Make sure at least 1 mesh exists
		m_MeshChanged.push_back(false);
	}
	if (m_TriangleBufferInfos.size() != m_Meshes.size()) // Recreate triangle buffer info for all
	{
		m_TriangleBufferInfos.resize(m_Meshes.size());
		for (uint i = 0; i < m_Meshes.size(); i++)
			m_TriangleBufferInfos[i] = m_Meshes[i]->triangles.getDescriptorBufferInfo();
	}
	else
	{
		for (uint i = 0; i < m_Meshes.size(); i++) // Update only those triangle buffer infos that have changed
			if (m_MeshChanged[i])
				m_TriangleBufferInfos[i] = m_Meshes[i]->triangles.getDescriptorBufferInfo();
	}

	shadeDescriptorSet->bind(cCOUNTERS, {m_Counters->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cCAMERA, {m_UniformCamera->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(
		cPATH_STATES, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(0, singleSize), m_CombinedStateBuffer[1]->getDescriptorBufferInfo(0, singleSize)});
	shadeDescriptorSet->bind(cPATH_ORIGINS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(singleSize, singleSize),
											 m_CombinedStateBuffer[1]->getDescriptorBufferInfo(singleSize, singleSize)});
	shadeDescriptorSet->bind(cPATH_DIRECTIONS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(2 * singleSize, singleSize),
												m_CombinedStateBuffer[1]->getDescriptorBufferInfo(2 * singleSize, singleSize)});
	shadeDescriptorSet->bind(cPATH_THROUGHPUTS, {m_CombinedStateBuffer[0]->getDescriptorBufferInfo(3 * singleSize, singleSize),
												 m_CombinedStateBuffer[1]->getDescriptorBufferInfo(3 * singleSize, singleSize)});
	shadeDescriptorSet->bind(cPOTENTIAL_CONTRIBUTIONS, {m_PotentialContributionBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cMATERIALS, {m_Materials->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cSKYBOX, {m_SkyboxImage->getDescriptorImageInfo()});
	shadeDescriptorSet->bind(cTRIANGLES, m_TriangleBufferInfos);
	shadeDescriptorSet->bind(cINVERSE_TRANSFORMS, {m_InvTransformsBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cTEXTURE_RGBA32, {m_RGBA32Buffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cTEXTURE_RGBA128, {m_RGBA128Buffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->bind(cBLUENOISE, {m_BlueNoiseBuffer->getDescriptorBufferInfo()});
	shadeDescriptorSet->updateSetContents();
	finalizeDescriptorSet->bind(fACCUMULATION_BUFFER, {m_AccumulationBuffer->getDescriptorBufferInfo()});
	finalizeDescriptorSet->bind(fUNIFORM_CONSTANTS, {m_UniformFinalizeParams->getDescriptorBufferInfo()});
	finalizeDescriptorSet->bind(fOUTPUT, {m_InteropTexture->getDescriptorImageInfo()});
	finalizeDescriptorSet->updateSetContents();
}

rfw::RenderContext *createRenderContext() { return new vkrtx::Context(); }

void destroyRenderContext(rfw::RenderContext *ptr)
{
	ptr->cleanup();
	delete ptr;
}
