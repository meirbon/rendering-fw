//
// Created by Mèir Noordermeer on 29/10/2019.
//

#ifndef RENDERINGFW_METALRAST_SRC_CONTEXT_H
#define RENDERINGFW_METALRAST_SRC_CONTEXT_H

#include <utils/LibExport.h>
#include <RenderContext.h>
#include <ContextExport.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3native.h>
#import <QuartzCore/QuartzCore.h>

#import <Metal/Metal.h>
#include <objc/objc.h>
#import <simd/simd.h>

class Context : public rfw::RenderContext
{
  public:
	Context();
	~Context();

	[[nodiscard]] std::vector<rfw::RenderTarget> getSupportedTargets() const override;

	// Initialization methods, by default these throw to indicate the chosen rendercontext does not support the
	// specified target
	void init(std::shared_ptr<rfw::utils::Window> &window) override;
	void init(GLuint *glTextureID, uint width, uint height) override;

	void cleanup() override;
	void render_frame(const rfw::Camera &camera, rfw::RenderStatus status) override;
	void set_materials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) override;
	void set_textures(const std::vector<rfw::TextureData> &textures) override;
	void set_mesh(size_t index, const rfw::Mesh &mesh) override;
	void set_instance(size_t i, size_t meshIdx, const mat4 &transform, const mat3 &inverse_transform) override;
	void set_sky(const std::vector<glm::vec3> &pixels, size_t width, size_t height) override;
	void set_lights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
					const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights) override;
	void get_probe_results(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const override;
	[[nodiscard]] rfw::AvailableRenderSettings get_settings() const override;
	void set_setting(const rfw::RenderSetting &setting) override;
	void update() override;
	void set_probe_index(glm::uvec2 probePos) override;
	rfw::RenderStats get_stats() const override;

  private:
	id<MTLDevice> m_Device;
	std::shared_ptr<rfw::utils::Window> m_GLFWWindow;
	NSWindow *m_Window;
	CAMetalLayer *m_Layer;

	int m_Width, m_Height;
	float m_AspectRatio;

	id<MTLLibrary> m_Library;
	id<MTLFunction> m_VertexFunction;
	id<MTLFunction> m_FragmentFunction;
	id<MTLCommandQueue> m_CommandQueue;
	id<MTLRenderPipelineState> m_RenderPipelineState;
};

#endif // RENDERINGFW_METALRAST_SRC_CONTEXT_H
