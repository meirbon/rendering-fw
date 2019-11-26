//
// Created by Mèir Noordermeer on 23/11/2019.
//

#include "Context.h"

#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>

using namespace rfw;

rfw::RenderContext *createRenderContext() { return new Context(); }

void destroyRenderContext(rfw::RenderContext *ptr) { ptr->cleanup(), delete ptr; }

Context::~Context()
{
	for (GLMesh *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();
	m_Textures.clear();

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
		CheckGL();
	}

	if (m_FboID)
	{
		glDeleteFramebuffers(1, &m_FboID);
		glDeleteRenderbuffers(1, &m_RboID);

		m_FboID = 0;
		m_RboID = 0;
	}

	m_Width = width;
	m_Height = height;

	m_TargetID = *glTextureID;
	glGenFramebuffers(1, &m_FboID);
	glGenRenderbuffers(1, &m_RboID);

	glBindRenderbuffer(GL_RENDERBUFFER, m_RboID);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_Width, m_Height);
	glBindRenderbuffer(GL_RENDERBUFFER, 0);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_TargetID, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, m_RboID);
	CheckGL();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	assert(m_TargetID);
	assert(m_FboID);
	assert(m_RboID);
}

void Context::cleanup()
{
	if (m_FboID)
	{
		glDeleteFramebuffers(1, &m_FboID), m_FboID = 0;
		glDeleteRenderbuffers(1, &m_RboID), m_RboID = 0;
	}

	for (auto *mesh : m_Meshes)
		delete mesh;
	m_Meshes.clear();

	m_Textures.clear();

	delete m_Materials;
	m_Materials = nullptr;
}

void Context::renderFrame(const rfw::Camera &camera, rfw::RenderStatus status)
{
	CheckGL();
	glBindFramebuffer(GL_FRAMEBUFFER, m_FboID);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glClearColor(0, 0, 0, 1.0f);

	m_SimpleShader->bind();
	auto matrix = camera.getMatrix(0.1f, 1e16f);
	const auto matrix3x3 = mat3(matrix);
	matrix = glm::scale(matrix, vec3(1, -1, 1));
	m_SimpleShader->setUniform("CamMatrix", matrix);
	m_SimpleShader->setUniform("CamMatrix3x3", matrix3x3);

	for (int i = 0; i < m_Textures.size(); i++)
	{
		m_Textures.at(i).bind(i);

		char buffer[128];
		sprintf(buffer, "textures[%i]", i);
		m_SimpleShader->setUniform(buffer, m_TextureBindings.at(i));
	}

	CheckGL();
	assert(m_Materials);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, m_Materials->getID());
	CheckGL();

	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		const GLMesh *mesh = m_Meshes.at(i);
		const std::vector<glm::mat4> &instance = m_Instances.at(i);
		const std::vector<glm::mat4> &inverseInstance = m_InverseInstances.at(i);

		if (instance.empty())
			continue;

		mesh->vao.bind();

		int sj = 0;
		if (instance.size() < 32)
			sj = 1;
		else
			sj = static_cast<int>(instance.size() / 32);

		for (int j = 0; j < sj; j++)
		{
			const auto offset = j * 32;
			const auto count = min(32, int(instance.size()) - offset);

			// Update instance matrices
			m_SimpleShader->setUniform("InstanceMatrices", instance.data() + offset, count, false);
			m_SimpleShader->setUniform("InverseMatrices", inverseInstance.data() + offset, count, false);
			mesh->draw(*m_SimpleShader, count);
		}
	}

	m_SimpleShader->unbind();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);
}

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors)
{
	delete m_Materials;
	m_Materials = new utils::Buffer<DeviceMaterial, GL_SHADER_STORAGE_BUFFER, GL_STATIC_READ>();
	m_Materials->setData(materials);
	CheckGL();

	const auto mats = m_Materials->read();

	printf("hi");
}

void Context::setTextures(const std::vector<rfw::TextureData> &textures)
{
	GLint value;
	glGetIntegerv(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS, &value);

	if (value < textures.size())
	{
		FAILURE("Too many textures supplied, maximum supported by current GPU: %i", value);
	}

	m_Textures.clear();
	m_Textures.resize(textures.size());
	m_TextureBindings.resize(textures.size());

	for (int i = 0; i < textures.size(); i++)
	{
		const auto &tex = textures.at(i);
		auto &glTex = m_Textures.at(i);

		if (tex.type == TextureData::FLOAT4)
			glTex.setData(static_cast<vec4 *>(tex.data), tex.width, tex.height);
		else
			glTex.setData(static_cast<uint *>(tex.data), tex.width, tex.height);

		glTex.generateMipMaps();
		glTex.bind(i);
		m_TextureBindings.at(i) = i;
	}

	CheckGL();
}

void Context::setMesh(size_t index, const rfw::Mesh &mesh)
{
	if (m_Meshes.size() <= index)
		m_Meshes.push_back(new GLMesh());
	CheckGL();
	m_Meshes.at(index)->setMesh(mesh);
	CheckGL();
}

void Context::setInstance(size_t i, size_t meshIdx, const mat4 &transform)
{
	if (m_InstanceGeometry.size() <= i)
	{
		m_InstanceGeometry.push_back(int(meshIdx));
		m_InstanceMatrices.push_back(transform);
	}

	m_InstanceGeometry.at(i) = int(meshIdx);
	m_InstanceMatrices.at(i) = transform;
}

void Context::setSkyDome(const std::vector<glm::vec3> &pixels, size_t width, size_t height) {}

void Context::setLights(rfw::LightCount lightCount, const rfw::DeviceAreaLight *areaLights, const rfw::DevicePointLight *pointLights,
						const rfw::DeviceSpotLight *spotLights, const rfw::DeviceDirectionalLight *directionalLights)
{
}

void Context::getProbeResults(unsigned int *instanceIndex, unsigned int *primitiveIndex, float *distance) const {}

rfw::AvailableRenderSettings Context::getAvailableSettings() const { return rfw::AvailableRenderSettings(); }

void Context::setSetting(const rfw::RenderSetting &setting) {}

void Context::update()
{
	m_Instances.resize(m_Meshes.size());
	m_InverseInstances.resize(m_Meshes.size());
	for (auto &i : m_Instances)
		i.clear();
	for (auto &i : m_InverseInstances)
		i.clear();

	for (int i = 0, s = static_cast<int>(m_InstanceGeometry.size()); i < s; i++)
	{
		m_Instances.at(m_InstanceGeometry.at(i)).push_back(m_InstanceMatrices.at(i));
		m_InverseInstances.at(m_InstanceGeometry.at(i)).push_back(inverse(m_InstanceMatrices.at(i)));
	}
}

void Context::setProbePos(glm::uvec2 probePos) {}

rfw::RenderStats Context::getStats() const { return rfw::RenderStats(); }
