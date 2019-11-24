//
// Created by Mèir Noordermeer on 23/11/2019.
//

#include "Context.h"

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context()
{
	for (GLMesh *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();

	delete m_SimpleShader;
}

std::vector<rfw::RenderTarget> Context::getSupportedTargets() const { return {rfw::RenderTarget::OPENGL_TEXTURE}; }

void Context::init(std::shared_ptr<rfw::utils::Window> &window) { throw std::runtime_error("Not supported (yet)."); }

void Context::init(GLuint *glTextureID, uint width, uint height)
{
	if (!m_InitializedGlew)
	{
		auto error = glewInit();
		if (error != GLEW_NO_ERROR)
			throw std::runtime_error("Could not init GLEW.");
		m_InitializedGlew = true;

		m_SimpleShader = new utils::GLShader("glshaders/simple.vert", "glshaders/simple.frag");
	}

	m_Width = width;
	m_Height = height;

	m_TargetID = *glTextureID;
	glGenFramebuffers(1, &m_FboID);
	glGenRenderbuffers(1, &m_RboID);

	glBindRenderbuffer(GL_RENDERBUFFER, m_RboID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_Width, m_Height);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_RboID);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	CheckGL();

	assert(m_TargetID);
	assert(m_FboID);
	assert(m_RboID);
}

void Context::cleanup()
{
	glDeleteFramebuffers(1, &m_FboID);
	glDeleteRenderbuffers(1, &m_RboID);
}

void Context::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	CheckGL();
	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	CheckGL();
	glClearColor(0, 0, 0, 1.0f);
	CheckGL();
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	CheckGL();

#if 1
	m_SimpleShader->bind();
	const auto matrix = camera.getMatrix();
	m_SimpleShader->setUniform("CamMatrix", matrix);

	CheckGL();

	for (auto *mesh : m_Meshes)
	{
		assert(mesh->VAO);

		m_SimpleShader->bind();
		glBindVertexArray(mesh->VAO);
		CheckGL();
		if (mesh->hasIndices)
		{
			mesh->indexBuffer.bind();
			CheckGL();
			glDrawElements(GL_TRIANGLES, mesh->indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr);
			CheckGL();
		}
		else
		{
			glDrawArrays(GL_TRIANGLES, 0, mesh->vertexBuffer.getCount());
			CheckGL();
		}
	}

	m_SimpleShader->unbind();
#endif
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void Context::setTextures(const std::vector<rfw::TextureData> &textures) {}

void Context::setMesh(size_t index, const rfw::Mesh &mesh)
{
	if (m_Meshes.size() <= index)
		m_Meshes.push_back(new GLMesh());
	CheckGL();
	m_Meshes.at(index)->setMesh(mesh);
	CheckGL();
}

void Context::setInstance(size_t i, size_t meshIdx, const mat4 &transform) {}

void Context::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

void Context::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
						const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights)
{
}

void Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::getAvailableSettings() const { return rfw::AvailableRenderSettings(); }

void Context::setSetting(const rfw::RenderSetting &setting) {}

void Context::update() {}

void Context::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats Context::getStats() const { return rfw::RenderStats(); }
