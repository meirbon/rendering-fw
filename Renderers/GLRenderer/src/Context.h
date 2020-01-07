#pragma once

#include <RenderContext.h>
#include <ContextExport.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>

#include <GL/glew.h>

#include <utils/gl/CheckGL.h>
#include <utils/gl/GLBuffer.h>
#include <utils/gl/GLShader.h>
#include <utils/gl/GLTexture.h>
#include <utils/gl/GLTextureArray.h>

#include "Mesh.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"
#include "../../CPURT/src/BVH/AABB.h"

namespace rfw
{

class Context : public RenderContext
{
	~Context() override;
	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override;

	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void renderFrame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void setTextures(const std::vector<rfw::TextureData> &textures) override;
	void setMesh(size_t index, const rfw::Mesh &mesh) override;
	void setInstance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
				   const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights) override;
	void getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	rfw::AvailableRenderSettings getAvailableSettings() const override;
	void setSetting(const rfw::RenderSetting &setting) override;
	void update() override;
	void setProbePos(glm::uvec2 probePos) override;
	rfw::RenderStats getStats() const override;

  private:
	utils::GLTexture m_Skybox;
	glm::vec3 m_Ambient = glm::vec3(0.15f);

	void setLights(utils::GLShader *shader);

	std::vector<std::vector<glm::mat4>> m_Instances;
	std::vector<std::vector<glm::mat4>> m_InverseInstances;

	std::vector<int> m_InstanceGeometry;
	std::vector<glm::mat4> m_InstanceMatrices;
	std::vector<glm::mat4> m_InverseInstanceMatrices;

	std::vector<utils::GLTexture> m_Textures;
	std::vector<GLint> m_TextureBindings;
	std::vector<DeviceMaterial> m_Materials;

	LightCount m_LightCount;
	std::vector<PointLight> m_PointLights;
	std::vector<AreaLight> m_AreaLights;
	std::vector<DirectionalLight> m_DirectionalLights;
	std::vector<SpotLight> m_SpotLights;

	utils::GLShader *m_CurrentShader;

	utils::GLShader *m_ColorShader;
	utils::GLShader *m_NormalShader;
	utils::GLShader *m_SimpleShader;
	std::vector<GLMesh *> m_Meshes;
	bool m_InitializedGlew = false;
	GLuint m_TargetID, m_FboID, m_RboID;
	GLuint m_Width, m_Height;
	rfw::RenderStats m_RenderStats;
};

} // namespace rfw
