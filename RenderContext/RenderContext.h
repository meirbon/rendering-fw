//
// Created by Mèir Noordermeer on 24/08/2019.
//

#ifndef RENDERING_FW_SRC_RENDERCONTEXT_H
#define RENDERING_FW_SRC_RENDERCONTEXT_H

#include <vector>
#include <memory>

#include <MathIncludes.h>

#include <Structures.h>

#include <utils/Window.h>

#include <Camera.h>

namespace rfw
{

enum RenderStatus
{
	Reset = 0,
	Converge = 1,
};

// TODO: Not used currently, but intention is to support multiple types of render targets
// Not every rendercontext needs to support every rendertarget, instead, supported render targets should be queried
enum RenderTarget
{
	VULKAN_TEXTURE,
	OPENGL_TEXTURE,
	METAL_TEXTURE,
	BUFFER,
	WINDOW
};

struct AvailableRenderSettings
{
	std::vector<std::string> settingKeys;
	std::vector<std::vector<std::string>> settingValues;
};

struct RenderSetting
{
	RenderSetting(const std::string &key, std::string val) : name(key), value(std::move(val)) {}

	std::string name;
	std::string value;
};

struct RenderStats
{
	RenderStats() { clear(); }
	void clear() { memset(this, 0, sizeof(RenderStats)); }

	float primaryTime;
	unsigned int primaryCount;

	float secondaryTime;
	unsigned int secondaryCount;

	float deepTime;
	unsigned int deepCount;

	float shadowTime;
	unsigned int shadowCount;

	float shadeTime;
	float finalizeTime;
};

class RenderContext
{
  public:
	RenderContext() = default;
	virtual ~RenderContext() = default;

	[[nodiscard]] virtual std::vector<rfw::RenderTarget> getSupportedTargets() const = 0;

	// Initialization methods, by default these throw to indicate the chosen rendercontext does not support the
	// specified target
	virtual void init(std::shared_ptr<rfw::utils::Window> &window)
	{
		throw std::runtime_error("RenderContext does not support given target type.");
	};
	virtual void init(GLuint *glTextureID, uint width, uint height)
	{
		throw std::runtime_error("RenderContext does not support given target type.");
	};

	virtual void cleanup() = 0;
	virtual void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) = 0;
	virtual void setMaterials(const std::vector<rfw::DeviceMaterial> &materials,
							  const std::vector<rfw::MaterialTexIds> &texDescriptors) = 0;
	virtual void setTextures(const std::vector<rfw::TextureData> &textures) = 0;
	virtual void setMesh(size_t index, const rfw::Mesh &mesh) = 0;
	virtual void setInstance(size_t i, size_t meshIdx, const mat4 &transform) = 0;
	virtual void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) = 0;
	virtual void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
						   const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
						   const rfw::DeviceDirectionalLight *directionalLights) = 0;
	virtual void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const = 0;
	virtual rfw::AvailableRenderSettings getAvailableSettings() const = 0;
	virtual void setSetting(const rfw::RenderSetting &setting) = 0;
	virtual void update() = 0;
	virtual void setProbePos(glm::uvec2 probePos) = 0;
	virtual rfw::RenderStats getStats() const = 0;
};

} // namespace rfw

#endif // RENDERING_FW_SRC_RENDERCONTEXT_H
