//
// Created by Mèir Noordermeer on 29/10/2019.
//
#include "RenderSystem.h"
#include "RenderSystemCAPI.h"

rfwRenderSystemHandle createRenderSystem()
{
	rfwRenderSystemHandle handle{};
	handle.instance = new rfw::RenderSystem();
	return handle;
}

void cleanupRenderSystem(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	delete handle;
}

void loadRenderAPI(rfwRenderSystemHandle *handle, const char *name)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->loadRenderAPI(name);
}

void unloadRenderAPI(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->unloadRenderAPI();
}

void setTarget(rfwRenderSystemHandle *handle, GLuint *textureID, unsigned int width, unsigned int height)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->setTarget(textureID, width, height);
}

void setSkybox(rfwRenderSystemHandle *handle, const char *filename)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->setSkybox(filename);
}

void synchronize(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->synchronize();
}

void updateAnimationsTo(rfwRenderSystemHandle *handle, float timeInSeconds)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->updateAnimationsTo(timeInSeconds);
}

rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, int material)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	auto ref = rs->addObject(fileName, material);
	rfwGeometryReference r;
	r.index = ref.getIndex();
	return r;
}

rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, bool normalize, int material)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	auto ref = rs->addObject(fileName, normalize, material);
	rfwGeometryReference r;
	r.index = ref.getIndex();
	return r;
}

rfwGeometryReference addObject(rfwRenderSystemHandle *handle, const char *fileName, bool normalize,
							   const float *preTransform, int material)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	glm::mat4 matrix = glm::mat4(1.0f);
	if (preTransform != nullptr)
		memcpy(value_ptr(matrix), preTransform, 16 * sizeof(float));

	auto ref = rs->addObject(fileName, normalize, matrix, material);
	rfwGeometryReference r;
	r.index = ref.getIndex();
	return r;
}

rfwGeometryReference addQuad(rfwRenderSystemHandle *handle, vec3 N, vec3 pos, float width, float height,
							 unsigned int material)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	const glm::vec3 gN = {N.x, N.y, N.z};
	const glm::vec3 gPos = {pos.x, pos.y, pos.z};
	auto ref = rs->addQuad(gN, gPos, width, height, material);
	rfwGeometryReference r;
	r.index = ref.getIndex();
	return r;
}

rfwInstanceReference addInstance(rfwRenderSystemHandle *handle, rfwGeometryReference geometry, const rfwvec3 &scaling,
								 const rfwvec3 &translation, float degrees, const rfwvec3 &axes)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	rfw::GeometryReference ref = rs->getGeometryReference(geometry.index);
	auto ir = rs->addInstance(ref, {scaling.x, scaling.y, scaling.z}, {translation.x, translation.y, translation.z},
							   degrees, {axes.x, axes.y, axes.z});
	rfwInstanceReference r;
	r.index = ir.getIndex();
	return r;
}

void updateInstance(rfwRenderSystemHandle *handle, rfwInstanceReference instanceRef, const float *transform)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	glm::mat4 matrix;
	memcpy(value_ptr(matrix), transform, 16 * sizeof(float));

	auto ref = rs->getInstanceReference(instanceRef.index);
	rs->updateInstance(ref, matrix);
}

void setAnimationTime(rfwRenderSystemHandle *handle, rfwGeometryReference geometry, float timeInSeconds)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	rfw::GeometryReference ref = rs->getGeometryReference(geometry.index);
	rs->setAnimationTime(ref, timeInSeconds);
}

HostMaterial getMaterial(rfwRenderSystemHandle *handle, size_t index)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	const auto mat = rs->getMaterial(index);

	HostMaterial cMat;
	memcpy(&cMat, &mat, sizeof(HostMaterial));
	return cMat;
}

void setMaterial(rfwRenderSystemHandle *handle, size_t index, const HostMaterial &mat)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	rfw::HostMaterial material = rs->getMaterial(index);
	material.name = mat.name;
	material.color = {mat.color.x, mat.color.y, mat.color.z};
	material.absorption = {mat.absorption.x, mat.absorption.y, mat.absorption.z};
	material.metallic = mat.metallic;
	material.subsurface = mat.subsurface;
	material.specular = mat.specular;
	material.roughness = mat.roughness;
	material.specularTint = mat.specularTint;
	material.anisotropic = mat.anisotropic;
	material.sheen = mat.sheen;
	material.sheenTint = mat.sheenTint;
	material.clearcoat = mat.clearcoat;
	material.clearcoatGloss = mat.clearcoatGloss;
	material.transmission = mat.transmission;
	material.eta = mat.eta;
	material.custom0 = mat.custom0;
	material.custom1 = mat.custom1;
	material.custom2 = mat.custom2;
	material.custom3 = mat.custom3;

	rs->setMaterial(index, material);
}

unsigned int addMaterial(rfwRenderSystemHandle *handle, const rfwvec3 &color, float roughness)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	return rs->addMaterial({color.x, color.y, color.z}, roughness);
}

#define RESET 0
#define CONVERGE 1

void renderFrame(rfwRenderSystemHandle *handle, const Camera &camera, unsigned int status)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	assert(status >= 0 && status <= 1);

	status = clamp(status, 0u, 1u);
	rfw::Camera cam;
	memcpy(&cam, &camera, sizeof(Camera));
	rs->renderFrame(cam, (rfw::RenderStatus)status);
}

LightReference addPointLight(rfwRenderSystemHandle *handle, const rfwvec3 &position, float energy,
							 const rfwvec3 &radiance)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	auto ref = rs->addPointLight({position.x, position.y, position.z}, energy, {radiance.x, radiance.y, radiance.z});
	LightReference reference{};
	reference.index = ref.getIndex();
	reference.type = LIGHT_TYPE_POINT;
	return reference;
}

LightReference addSpotLight(rfwRenderSystemHandle *handle, const vec3 &position, float cosInner, const vec3 &radiance,
							float cosOuter, float energy, const vec3 &direction)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	auto ref = rs->addSpotLight({position.x, position.y, position.z}, cosInner, {radiance.x, radiance.y, radiance.z},
								cosOuter, energy, {direction.x, direction.y, direction.z});
	LightReference reference{};
	reference.index = ref.getIndex();
	reference.type = LIGHT_TYPE_SPOT;
	return reference;
}

LightReference addDirectionalLight(rfwRenderSystemHandle *handle, const rfwvec3 &direction, float energy,
								   const rfwvec3 &radiance)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	auto ref =
		rs->addDirectionalLight({direction.x, direction.y, direction.z}, energy, {radiance.x, radiance.y, radiance.z});
	LightReference reference{};
	reference.index = ref.getIndex();
	reference.type = LIGHT_TYPE_DIRECTIONAL;
	return reference;
}

void setPosition(rfwRenderSystemHandle *handle, const LightReference &reference, const rfwvec3 &position)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	switch (reference.type)
	{
	case (LIGHT_TYPE_AREA):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setPosition(ref, {position.x, position.y, position.z});
		break;
	}
	case (LIGHT_TYPE_POINT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setPosition(ref, {position.x, position.y, position.z});
		break;
	}
	case (LIGHT_TYPE_SPOT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setPosition(ref, {position.x, position.y, position.z});
		break;
	}
	case (LIGHT_TYPE_DIRECTIONAL):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setPosition(ref, {position.x, position.y, position.z});
		break;
	}
	}
}

void setRadiance(rfwRenderSystemHandle *handle, const LightReference &reference, const rfwvec3 &radiance)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	switch (reference.type)
	{
	case (LIGHT_TYPE_AREA):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setRadiance(ref, {radiance.x, radiance.y, radiance.z});
		break;
	}
	case (LIGHT_TYPE_POINT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setRadiance(ref, {radiance.x, radiance.y, radiance.z});
		break;
	}
	case (LIGHT_TYPE_SPOT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setRadiance(ref, {radiance.x, radiance.y, radiance.z});
		break;
	}
	case (LIGHT_TYPE_DIRECTIONAL):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setRadiance(ref, {radiance.x, radiance.y, radiance.z});
		break;
	}
	}
}

void setEnergy(rfwRenderSystemHandle *handle, const LightReference &reference, float energy)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	switch (reference.type)
	{
	case (LIGHT_TYPE_AREA):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setEnergy(ref, energy);
		break;
	}
	case (LIGHT_TYPE_POINT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setEnergy(ref, energy);
		break;
	}
	case (LIGHT_TYPE_SPOT):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setEnergy(ref, energy);
		break;
	}
	case (LIGHT_TYPE_DIRECTIONAL):
	{
		rfw::LightReference ref = rs->getPointLightReference(reference.index);
		rs->setEnergy(ref, energy);
		break;
	}
	}
}

AvailableRenderSettings getAvailableSettings(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	const auto availableSettings = rs->getAvailableSettings();

	AvailableRenderSettings settings{};
	settings.numKeys = (uint)availableSettings.settingKeys.size();
	if (!availableSettings.settingKeys.empty())
	{
		settings.settingKeys = new char *[settings.numKeys];
		for (uint i = 0; i < settings.numKeys; i++)
		{
			const auto key = availableSettings.settingKeys.at(i);

			settings.settingKeys[i] = new char[key.size()];
			memcpy(settings.settingKeys, key.data(), key.size() * sizeof(char));
		}
	}
	settings.numSettingValues = (uint)availableSettings.settingValues.size();
	if (!availableSettings.settingValues.empty())
	{
		settings.settingValues = new char *[settings.numSettingValues];
		for (uint i = 0; i < settings.numKeys; i++)
		{
			const auto value = availableSettings.settingValues.at(i);

			settings.settingValues[i] = new char[value.size()];
			memcpy(settings.settingKeys, value.data(), value.size() * sizeof(char));
		}
	}

	return settings;
}

void setSetting(rfwRenderSystemHandle *handle, const RenderSetting &setting)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);

	const rfw::RenderSetting rfwSetting = rfw::RenderSetting(setting.name, setting.value);
	rs->setSetting(rfwSetting);
}

rfwAABB calculateSceneBounds(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	auto aabb = rs->calculateSceneBounds();

	auto bounds = rfwAABB();
	bounds.mMin = {aabb.mMin.x, aabb.mMin.y, aabb.mMin.z};
	bounds.mMax = {aabb.mMax.x, aabb.mMax.y, aabb.mMax.z};
	return bounds;
}

void setProbeIndex(rfwRenderSystemHandle *handle, rfwuvec2 pixelIdx)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	rs->setProbeIndex({pixelIdx.x, pixelIdx.y});
}

rfwProbeResult getProbeResult(rfwRenderSystemHandle *handle)
{
	auto rs = reinterpret_cast<rfw::RenderSystem *>(handle->instance);
	assert(rs != nullptr);
	const auto result = rs->getProbeResult();
	rfwProbeResult res;
	res.distance = result.distance;
	res.instanceIdx = result.object.getIndex();
	res.materialIdx = result.materialIdx;
	return res;
}