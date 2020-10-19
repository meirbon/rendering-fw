#include "PCH.h"

rfw::RenderContext *createRenderContext() { return new rfw::CUDAContext(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

rfw::CUDAContext::~CUDAContext() {}

std::vector<rfw::RenderTarget> rfw::CUDAContext::get_supported_targets() const
{
	return {rfw::RenderTarget::OPENGL_TEXTURE};
}

void rfw::CUDAContext::init(std::shared_ptr<rfw::utils::Window> &window)
{
	throw std::runtime_error("Not supported (yet).");
}

void rfw::CUDAContext::init(GLuint *glTextureID, uint width, uint height)
{
	cudaFree(nullptr); // Initialize CUDA device
	CheckCUDA(cudaDeviceSynchronize());

	m_Width = width;
	m_Height = height;
	m_TargetID = *glTextureID;
	m_SampleIndex = 0;

	if (!m_Initialized)
	{
		if (!m_InitializedGlew)
		{
			auto error = glewInit();
			if (error != GLEW_NO_ERROR)
				throw std::runtime_error("Could not init GLEW.");
			m_InitializedGlew = true;
			CheckGL();
		}

		m_Counters = std::make_unique<CUDABuffer<Counters>>(1);

		setCounters(m_Counters->device_data());
		const auto blueNoiseBuffer = createBlueNoiseBuffer();
		m_BlueNoise = std::make_unique<CUDABuffer<uint>>(blueNoiseBuffer.size(), ON_DEVICE);
		m_BlueNoise->copy_to_device(blueNoiseBuffer);
		setBlueNoiseBuffer(m_BlueNoise->device_data());
		m_CameraView = std::make_unique<CUDABuffer<CameraView>>(1, ON_ALL);
		setCameraView(m_CameraView->device_data());

		setGeometryEpsilon(1e-5f);
		setClampValue(10.0f);
		m_Initialized = true;
	}

	m_Counters->data()[0].sampleIndex = 0;
	setScreenDimensions(m_Width, m_Height);
	resizeBuffers();
	setupTexture();

	m_ResetFrame = true;
}

void rfw::CUDAContext::cleanup() {}

void rfw::CUDAContext::render_frame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	glFinish();
	m_Stats.clear();
	Counters *counters = m_Counters->data();
	counters->probeIdx = uint(m_ProbePixel.x + m_ProbePixel.y * m_Width);

	const auto view = camera.get_view();
	m_CameraView->copy_to_device_async(&view, 1);

	if (status == Reset)
	{
		counters->sampleIndex = 0;
		m_Accumulator->clear_async();
		m_SampleIndex = 0;
	}
	m_Counters->copy_to_device_async();

	uint pathLength = 0;
	uint pathCount = m_Width * m_Height;

	InitCountersForExtend(pathCount, m_SampleIndex);
	auto timer = utils::Timer();
	CheckCUDA(intersectRays(Primary, pathLength, m_Width, m_Height));
	m_Stats.primaryCount = pathCount;
	CheckCUDA(cudaDeviceSynchronize());
	m_Stats.primaryTime = timer.elapsed();

	timer.reset();
	CheckCUDA(shadeRays(pathLength, pathCount));
	CheckCUDA(cudaDeviceSynchronize());
	m_Stats.shadeTime += timer.elapsed();

	m_Counters->copy_to_host();
	uint activePaths = counters->extensionRays;

	if (m_SampleIndex == 0)
	{
		m_ProbedInstance = counters->probedInstanceId;
		m_ProbedPrim = counters->probedPrimId;
		m_ProbedDistance = counters->probedDistance;
		m_ProbedPoint = counters->probedPoint;
	}

	while (activePaths > 0 && pathLength < MAX_PATH_LENGTH)
	{
		pathLength = pathLength + 1;
		if (counters->shadowRays > 0)
		{
			m_Stats.shadowCount += counters->shadowRays;

			timer.reset();
			intersectRays(Shadow, pathLength, counters->shadowRays);
			CheckCUDA(cudaDeviceSynchronize());
			m_Stats.shadowTime += timer.elapsed();
		}

		InitCountersSubsequent();
		CheckCUDA(cudaDeviceSynchronize());

		timer.reset();
		intersectRays(Secondary, pathLength, activePaths);
		if (pathLength == 1)
		{
			CheckCUDA(cudaDeviceSynchronize());
			m_Stats.secondaryCount += activePaths;
			m_Stats.secondaryTime += timer.elapsed();
		}
		else
		{
			CheckCUDA(cudaDeviceSynchronize());
			m_Stats.deepCount += activePaths;
			m_Stats.deepTime += timer.elapsed();
		}

		timer.reset();
		CheckCUDA(shadeRays(pathLength, activePaths));
		CheckCUDA(cudaDeviceSynchronize());
		m_Stats.shadeTime += timer.elapsed();

		m_Counters->copy_to_host();
		activePaths = counters->extensionRays;
	}

	m_SampleIndex++;
	CheckCUDA(cudaDeviceSynchronize());
	m_CUDASurface.bindSurface();

	CheckCUDA(blitBuffer(m_Width, m_Height, m_SampleIndex));
	m_CUDASurface.unbindSurface();

	counters->samplesTaken = m_SampleIndex;
	counters->activePaths = m_Width * m_Height;
	m_Counters->copy_to_device_async();
}

void rfw::CUDAContext::set_materials(const std::vector<rfw::DeviceMaterial> &materials,
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
		m_Materials = std::make_unique<CUDABuffer<DeviceMaterial>>(mats.size(), ON_DEVICE);

	m_Materials->copy_to_device_async(mats.data(), mats.size());

	::setMaterials(m_Materials->device_data());
}

void rfw::CUDAContext::set_textures(const std::vector<rfw::TextureData> &textures)
{
	m_TexDescriptors = textures;

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

			if (tex.type != TextureData::UINT)
				continue;

			assert((texelOffset + static_cast<size_t>(tex.texelCount)) <= uintTexs.size());
			m_TexDescriptors.at(i).texAddr = static_cast<uint>(texelOffset);

			memcpy(&uintTexs.at(texelOffset), tex.data, tex.texelCount * sizeof(uint));
			texelOffset += tex.texelCount;
		}
	}

	m_FloatTextures = std::make_unique<CUDABuffer<glm::vec4>>(floatTexs, ON_DEVICE);
	m_UintTextures = std::make_unique<CUDABuffer<uint>>(uintTexs, ON_DEVICE);

	setFloatTextures(m_FloatTextures->device_data());
	setUintTextures(m_UintTextures->device_data());
}

void rfw::CUDAContext::set_mesh(size_t index, const rfw::Mesh &mesh)
{
	while (index >= m_Meshes.size())
	{
		m_Meshes.push_back(std::make_unique<bvh::rfwMesh>());
		m_MeshVertices.push_back(std::make_unique<CUDABuffer<glm::vec4>>());
		m_MeshIndices.push_back(std::make_unique<CUDABuffer<glm::uvec3>>());

		m_MeshTriangles.push_back(std::make_unique<CUDABuffer<rfw::DeviceTriangle>>());
		m_MeshBVHs.push_back(std::make_unique<CUDABuffer<bvh::BVHNode>>());
		m_MeshMBVHs.push_back(std::make_unique<CUDABuffer<bvh::MBVHNode>>());
		m_MeshBVHPrimIndices.push_back(std::make_unique<CUDABuffer<uint>>());
	}

	std::unique_ptr<bvh::rfwMesh> &m = m_Meshes[index];

	m->set_geometry(mesh);

	if (m_MeshBVHs[index]->size() != m->bvh->instance.value().node_count)
	{
		m_MeshVertices[index] = std::make_unique<CUDABuffer<glm::vec4>>(m->vertexCount);
		m_MeshIndices[index] = std::make_unique<CUDABuffer<glm::uvec3>>(m->triangleCount);
		m_MeshTriangles[index] = std::make_unique<CUDABuffer<rfw::DeviceTriangle>>(m->triangleCount);
		m_MeshBVHs[index] = std::make_unique<CUDABuffer<bvh::BVHNode>>(m->bvh->instance.value().node_count, ON_DEVICE);
		m_MeshMBVHs[index] =
			std::make_unique<CUDABuffer<bvh::MBVHNode>>(m->mbvh->instance.value().node_count, ON_DEVICE);
		m_MeshBVHPrimIndices[index] =
			std::make_unique<CUDABuffer<unsigned int>>(m->bvh->instance.value().index_count, ON_DEVICE);
	}

	m_MeshVertices[index]->copy_to_device_async(m->vertices, m->vertexCount);
	if (m->indices)
		m_MeshIndices[index]->copy_to_device_async(m->indices, m->triangleCount);
	m_MeshTriangles[index]->copy_to_device_async((rfw::DeviceTriangle *)(m->triangles), m->triangleCount);
	m_MeshBVHs[index]->copy_to_device_async(reinterpret_cast<const rfw::bvh::BVHNode *>(m->bvh->instance.value().nodes),
											m->bvh->instance.value().node_count);
	m_MeshMBVHs[index]->copy_to_device_async(
		reinterpret_cast<const rfw::bvh::MBVHNode *>(m->mbvh->instance.value().nodes),
		m->mbvh->instance.value().node_count);
	m_MeshBVHPrimIndices[index]->copy_to_device_async(m->bvh->instance.value().indices,
													  m->bvh->instance.value().index_count);
}

void rfw::CUDAContext::set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform)
{
	if (i >= m_InstanceMeshIDs.size())
		m_InstanceMeshIDs.push_back(0);

	m_TopLevelBVH.set_instance(static_cast<int>(i), transform, m_Meshes[meshIdx].get(),
							   m_Meshes[meshIdx]->bvh->get_aabb());
	m_InstanceMeshIDs[i] = uint(meshIdx);
}

void rfw::CUDAContext::set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height)
{
	m_Skybox = std::make_unique<CUDABuffer<glm::vec3>>(width * height, ON_DEVICE);
	m_Skybox->copy_to_device_async(pixels.data(), width * height);
	setSkybox(m_Skybox->device_data());
	setSkyDimensions(static_cast<uint>(width), static_cast<uint>(height));
}

void rfw::CUDAContext::set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
								  const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
								  const rfw::DeviceDirectionalLight *directionalLights)
{
	CheckCUDA(cudaDeviceSynchronize());

	if (lightCount.areaLightCount > 0)
	{
		if (!m_AreaLights || m_AreaLights->size() < lightCount.areaLightCount)
		{
			m_AreaLights = std::make_unique<CUDABuffer<DeviceAreaLight>>(lightCount.areaLightCount, ON_DEVICE);
			setAreaLights(m_AreaLights->device_data());
		}
		m_AreaLights->copy_to_device_async(areaLights, lightCount.areaLightCount);
	}

	if (lightCount.pointLightCount > 0)
	{
		if (!m_PointLights || m_PointLights->size() < lightCount.pointLightCount)
		{
			m_PointLights = std::make_unique<CUDABuffer<DevicePointLight>>(lightCount.pointLightCount, ON_DEVICE);
			setPointLights(m_PointLights->device_data());
		}
		m_PointLights->copy_to_device_async(pointLights, lightCount.pointLightCount);
	}

	if (lightCount.spotLightCount)
	{
		if (!m_SpotLights || m_SpotLights->size() < lightCount.spotLightCount)
		{
			m_SpotLights = std::make_unique<CUDABuffer<DeviceSpotLight>>(lightCount.spotLightCount, ON_DEVICE);
			setSpotLights(m_SpotLights->device_data());
		}
		m_SpotLights->copy_to_device_async(spotLights, lightCount.spotLightCount);
	}

	if (lightCount.directionalLightCount > 0)
	{
		if (!m_DirectionalLights || m_DirectionalLights->size())
		{
			m_DirectionalLights = std::make_unique<CUDABuffer<DeviceDirectionalLight>>(
				directionalLights, lightCount.directionalLightCount, ON_DEVICE);
			setDirectionalLights(m_DirectionalLights->device_data());
		}
		m_DirectionalLights->copy_to_device_async(directionalLights, lightCount.directionalLightCount);
	}

	setLightCount(lightCount);
	CheckCUDA(cudaDeviceSynchronize());
}

void rfw::CUDAContext::get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex,
										 float *distance) const
{
	*instanceIndex = m_ProbedInstance;
	*primitiveIndex = m_ProbedPrim;
	*distance = m_ProbedDistance;
}

rfw::AvailableRenderSettings rfw::CUDAContext::get_settings() const { return {}; }

void rfw::CUDAContext::set_setting(const rfw::RenderSetting &setting) {}

void rfw::CUDAContext::update()
{
	auto build_handle = std::async([&]() { m_TopLevelBVH.construct_bvh(); });
	if (!m_InstanceDescriptors || m_InstanceDescriptors->size() < m_InstanceMeshIDs.size())
		m_InstanceDescriptors = std::make_unique<CUDABuffer<InstanceBVHDescriptor>>(m_InstanceMeshIDs.size(), ON_ALL);

	for (int i = 0, s = static_cast<int>(m_InstanceMeshIDs.size()); i < s; i++)
	{
		const auto meshID = m_InstanceMeshIDs[i];
		InstanceBVHDescriptor &desc = m_InstanceDescriptors->data()[i];
		desc.vertices = m_MeshVertices[meshID]->device_data();

		if (m_Meshes[meshID]->indices)
			desc.indices = m_MeshIndices[meshID]->device_data();
		else
			desc.indices = nullptr;

		desc.bvh = m_MeshBVHs[meshID]->device_data();
		desc.mbvh = m_MeshMBVHs[meshID]->device_data();
		desc.bvh_indices = m_MeshBVHPrimIndices[meshID]->device_data();

		desc.instance_transform = m_TopLevelBVH.get_instance_matrix(i).matrix;
		desc.inverse_transform = m_TopLevelBVH.get_inverse_matrix(i).matrix;
		desc.normal_transform = mat3x4(m_TopLevelBVH.get_normal_matrix(i).matrix);
		desc.triangles = m_MeshTriangles[meshID]->device_data();
	}

	m_InstanceDescriptors->copy_to_device_async();
	build_handle.get();

	if (!m_TopLevelCUDABVH || m_TopLevelCUDABVH->size() < m_TopLevelBVH.bvh.value().node_count)
		m_TopLevelCUDABVH = std::make_unique<CUDABuffer<bvh::BVHNode>>(m_TopLevelBVH.bvh.value().node_count, ON_DEVICE);
	if (!m_TopLevelCUDAMBVH || m_TopLevelCUDAMBVH->size() < m_TopLevelBVH.mbvh.value().node_count)
		m_TopLevelCUDAMBVH =
			std::make_unique<CUDABuffer<bvh::MBVHNode>>(m_TopLevelBVH.mbvh.value().node_count, ON_DEVICE);
	if (!m_TopLevelCUDAPrimIndices || m_TopLevelCUDAPrimIndices->size() < m_TopLevelBVH.bvh.value().index_count)
		m_TopLevelCUDAPrimIndices =
			std::make_unique<CUDABuffer<uint>>(m_TopLevelBVH.bvh.value().index_count, ON_DEVICE);
	if (!m_CUDAInstanceTransforms || m_CUDAInstanceTransforms->size() < m_TopLevelBVH.matrices.size())
	{
		m_CUDAInstanceTransforms = std::make_unique<CUDABuffer<glm::mat4>>(m_TopLevelBVH.matrices.size(), ON_DEVICE);
		m_CUDAInverseTransforms = std::make_unique<CUDABuffer<glm::mat4>>(m_TopLevelBVH.matrices.size(), ON_DEVICE);
	}

	m_TopLevelCUDABVH->copy_to_device_async(reinterpret_cast<const bvh::BVHNode *>(m_TopLevelBVH.bvh.value().nodes),
											m_TopLevelBVH.bvh.value().node_count);
	m_TopLevelCUDAMBVH->copy_to_device_async(reinterpret_cast<const bvh::MBVHNode *>(m_TopLevelBVH.mbvh.value().nodes),
											 m_TopLevelBVH.mbvh.value().node_count);
	m_TopLevelCUDAPrimIndices->copy_to_device_async(m_TopLevelBVH.bvh.value().indices,
													m_TopLevelBVH.bvh.value().index_count);

	m_CUDAInstanceTransforms->copy_to_device_async((glm::mat4 *)m_TopLevelBVH.matrices.data(),
												   m_TopLevelBVH.matrices.size());
	m_CUDAInverseTransforms->copy_to_device_async((glm::mat4 *)m_TopLevelBVH.inverse_matrices.data(),
												  m_TopLevelBVH.inverse_matrices.size());

	setTopLevelBVH(m_TopLevelCUDABVH->device_data());
	setTopLevelMBVH(m_TopLevelCUDAMBVH->device_data());
	setTopPrimIndices(m_TopLevelCUDAPrimIndices->device_data());
	setInstances(m_InstanceDescriptors->device_data());

	CheckCUDA(cudaDeviceSynchronize());
}

void rfw::CUDAContext::set_probe_index(glm::uvec2 probePos) { m_ProbePixel = probePos; }

rfw::RenderStats rfw::CUDAContext::get_stats() const { return m_Stats; }

void rfw::CUDAContext::resizeBuffers()
{
	const uint pixelCount = m_Width * m_Height;
	if (m_Accumulator && m_Accumulator->size() >= pixelCount)
		return;

	m_Accumulator = std::make_unique<CUDABuffer<glm::vec4>>(pixelCount, ON_DEVICE);
	m_PathStates = std::make_unique<CUDABuffer<glm::vec4>>(pixelCount * 2, ON_DEVICE);
	m_PathOrigins = std::make_unique<CUDABuffer<glm::vec4>>(pixelCount * 2, ON_DEVICE);
	m_PathDirections = std::make_unique<CUDABuffer<glm::vec4>>(pixelCount * 2, ON_DEVICE);
	m_PathThroughputs = std::make_unique<CUDABuffer<glm::vec4>>(pixelCount * 2, ON_DEVICE);
	m_ConnectData = std::make_unique<CUDABuffer<PotentialContribution>>(pixelCount, ON_DEVICE);

	setAccumulator(m_Accumulator->device_data());
	setStride(pixelCount);
	setPathStates(m_PathStates->device_data());
	setPathOrigins(m_PathOrigins->device_data());
	setPathDirections(m_PathDirections->device_data());
	setPathThroughputs(m_PathThroughputs->device_data());
	setPotentialContributions(m_ConnectData->device_data());
	setScreenDimensions(m_Width, m_Height);
}

void rfw::CUDAContext::setupTexture()
{
	glBindTexture(GL_TEXTURE_2D, m_TargetID);
	m_CUDASurface.setTexture(m_TargetID);
	m_CUDASurface.linkToSurface(getOutputSurfaceReference());
	glBindTexture(GL_TEXTURE_2D, 0);
}
