#include "Context.h"
#import <utils.h>

Context::Context()
{
	m_Device = (id<MTLDevice>)MTLCreateSystemDefaultDevice();

	if (!m_Device)
		throw std::runtime_error("Could not initialize Metal device.");
}

Context::~Context() {}

std::vector<rfw::RenderTarget> Context::getSupportedTargets() const { return {rfw::WINDOW}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window)
{
	m_GLFWWindow = window;
	m_Window = glfwGetCocoaWindow(window->getGLFW());
	m_Layer = [CAMetalLayer layer];
	m_Layer.device = m_Device;
	m_Layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
	m_Window.contentView.layer = m_Layer;
	m_Window.contentView.wantsLayer = YES;

	MTLCompileOptions *compileOptions = [[MTLCompileOptions new] autorelease];
	compileOptions.languageVersion = MTLLanguageVersion2_2;

	const auto source = rfw::utils::file::read("metalshaders/shaders.metal");

	NSError *compileError;
	m_Library = [m_Device newLibraryWithSource:[NSString stringWithUTF8String:source.data()]
									   options:compileOptions
										 error:&compileError];
	if (!m_Library)
		FAILURE("Could not create Metal library: %s", compileError);

	m_VertexFunction = [m_Library newFunctionWithName:@"v_simple"];
	m_FragmentFunction = [m_Library newFunctionWithName:@"f_simple"];

	m_CommandQueue = [m_Device newCommandQueue];
	assert(m_CommandQueue);
	MTLRenderPipelineDescriptor *rpd = [[MTLRenderPipelineDescriptor new] autorelease];
	rpd.vertexFunction = m_VertexFunction;
	rpd.fragmentFunction = m_FragmentFunction;
	rpd.colorAttachments[0].pixelFormat = m_Layer.pixelFormat;
	m_RenderPipelineState = [m_Device newRenderPipelineStateWithDescriptor:rpd error:nullptr];
	assert(m_RenderPipelineState);

	m_Width = window->getFramebufferWidth();
	m_Height = window->getFramebufferHeight();
	m_AspectRatio = float(m_Width) / float(m_Height);
	m_Layer.drawableSize = CGSizeMake(m_Width, m_Height);
}

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	throw std::runtime_error("GL targets not supported yet.");
}

void Context::cleanup() {}

void Context::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	id<CAMetalDrawable> drawable = [m_Layer nextDrawable];
	id<MTLCommandBuffer> cmdBuffer = [m_CommandQueue commandBuffer];

	MTLRenderPassDescriptor *rpd = [[MTLRenderPassDescriptor new] autorelease];
	MTLRenderPassColorAttachmentDescriptor *cd = rpd.colorAttachments[0];
	cd.texture = drawable.texture;
	cd.loadAction = MTLLoadActionClear;
	cd.clearColor = MTLClearColorMake(0.0, 0.0, 0.0, 1.0);
	cd.storeAction = MTLStoreActionStore;
	id<MTLRenderCommandEncoder> rce = [cmdBuffer renderCommandEncoderWithDescriptor:rpd];
	[rce setRenderPipelineState:m_RenderPipelineState];

	const float vertices[] = {0, 0, 0, 1, -1, 1, 0, 1, 1, 1, 0, 1};

	[rce setVertexBytes:vertices length:3 * sizeof(simd_float4) atIndex:0];
	[rce drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];

	[rce endEncoding];
	[cmdBuffer presentDrawable:drawable];
	[cmdBuffer commit];
}

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials,
						   const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
}

void Context::setTextures(const std::vector<rfw::TextureData> &textures) {}

void Context::setMesh(size_t index, const rfw::Mesh &mesh) {}

void Context::setInstance(size_t i, size_t meshIdx, const mat4 &transform) {}

void Context::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

void Context::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights,
						const rfw::DevicePointLight *pointLights, const rfw::DeviceSpotLight *spotLights,
						const rfw::DeviceDirectionalLight *directionalLights)
{
}

void Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::getAvailableSettings() const { return rfw::AvailableRenderSettings(); }

void Context::setSetting(const rfw::RenderSetting &setting) {}

void Context::update() {}

void Context::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats Context::getStats() const { return rfw::RenderStats(); }

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr)
{
	ptr->cleanup();
	delete ptr;
}