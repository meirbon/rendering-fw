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
	glDeleteFramebuffers(1, &m_FboID);
	glDeleteRenderbuffers(1, &m_RboID);
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
	auto matrix = camera.getMatrix(0.1f, 1e2f);
	const auto matrix3x3 = mat3(matrix);
	matrix = glm::scale(matrix, vec3(1, -1, 1));
	m_SimpleShader->setUniform("CamMatrix", matrix);
	m_SimpleShader->setUniform("CamMatrix3x3", matrix3x3);

	CheckGL();

	for (int i = 0, s = static_cast<int>(m_Instances.size()); i < s; i++)
	{
		int sj = 0;
		if (m_Instances.at(i).size() > 0 && m_Instances.at(i).size() < 32)
			sj = 1;
		else
			sj = static_cast<int>(m_Instances.at(i).size() / 32);

		for (int j = 0; j < sj; j++)
		{
			const auto offset = j * 32;
			const auto count = min(32, int(m_Instances.at(i).size()) - offset);

			// Update instance matrices
			m_SimpleShader->setUniform("InstanceMatrices", m_Instances.at(i).data() + offset, count, false);
			m_SimpleShader->setUniform("InverseMatrices", m_InverseInstances.at(i).data() + offset, count, false);

			const GLMesh *mesh = m_Meshes.at(i);
			mesh->vao.bind();

			if (mesh->hasIndices)
			{
				mesh->indexBuffer.bind();
				glDrawElementsInstanced(GL_TRIANGLES, mesh->indexBuffer.getCount(), GL_UNSIGNED_INT, nullptr, (GLsizei)m_Instances.at(i).size());
			}
			else
			{
				glDrawArraysInstanced(GL_TRIANGLES, 0, mesh->vertexBuffer.getCount(), (GLsizei)m_Instances.at(i).size());
			}

			CheckGL();
		}
	}

	m_SimpleShader->unbind();

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDisable(GL_DEPTH_TEST);
}

void Context::setMaterials(const std::vector<rfw::DeviceMaterial> &materials, const std::vector<rfw::MaterialTexIds> &texDescriptors) {}

void Context::setTextures(const std::vector<rfw::TextureData> &textures)
{
	m_Textures.clear();
	m_Textures.resize(textures.size());

	for (int i = 0; i < m_Textures.size(); i++)
	{
		const auto &tex = textures.at(i);
		auto &glTex = m_Textures.at(i);
		if (tex.type == TextureData::FLOAT4)
			glTex.setData((vec4 *)tex.data, tex.width, tex.height, 0);
		else // tex.type == TextureData::UINT
			glTex.setData((uint *)tex.data, tex.width, tex.height, 0);

		glTex.generateMipMaps();
	}
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
