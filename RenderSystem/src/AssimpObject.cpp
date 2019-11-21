#include "AssimpObject.h"

#include <MathIncludes.h>

#include <array>
#include <cmath>

#include <assimp/cimport.h>
#include <assimp/matrix4x4.h>
#include <assimp/matrix3x3.h>
#include <assimp/vector3.h>
#include "Settings.h"
#include "utils/Logger.h"
#include "utils/Timer.h"

namespace rfw
{

static_assert(sizeof(DeviceTriangle) == sizeof(Triangle));
static_assert(sizeof(Material) == sizeof(DeviceMaterial));
static_assert(sizeof(rfw::DeviceMaterial) == sizeof(rfw::Material));

namespace utils
{
struct Timer;
}
} // namespace rfw

using namespace rfw;
using namespace glm;

glm::vec3 operator*(const glm::vec3 &a, const aiVector3D &b) { return glm::vec3(a.x * b.x, a.y * b.y, a.y * b.z); }
glm::vec3 operator/(const glm::vec3 &a, const aiVector3D &b) { return glm::vec3(a.x / b.x, a.y / b.y, a.y / b.z); }
glm::vec3 operator+(const glm::vec3 &a, const aiVector3D &b) { return glm::vec3(a.x + b.x, a.y + b.y, a.y + b.z); }
glm::vec3 operator-(const glm::vec3 &a, const aiVector3D &b) { return glm::vec3(a.x - b.x, a.y - b.y, a.y - b.z); }

glm::vec3 operator*(const aiVector3D &b, const glm::vec3 &a) { return glm::vec3(a.x * b.x, a.y * b.y, a.y * b.z); }
glm::vec3 operator/(const aiVector3D &b, const glm::vec3 &a) { return glm::vec3(a.x / b.x, a.y / b.y, a.y / b.z); }
glm::vec3 operator+(const aiVector3D &b, const glm::vec3 &a) { return glm::vec3(a.x + b.x, a.y + b.y, a.y + b.z); }
glm::vec3 operator-(const aiVector3D &b, const glm::vec3 &a) { return glm::vec3(a.x - b.x, a.y - b.y, a.y - b.z); }

AssimpObject::AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const mat4 &matrix, bool normalize, int material)
	: m_File(filename.data()), m_ID(ID)
{
	std::string directory;
	const size_t last_slash_idx = filename.rfind('/');
	if (std::string::npos != last_slash_idx)
		directory = filename.substr(0, last_slash_idx);

	Assimp::Importer importer{};
	importer.SetPropertyBool(AI_CONFIG_PP_PTV_NORMALIZE, normalize);

	const glm::mat4 identity = glm::mat4(1.0f);
	bool hasTransform = false;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			if (matrix[i][j] != identity[i][j])
			{
				hasTransform = true;
				break;
			}
		}
	}

	const mat3 matrix3x3 = mat3(matrix);

	const aiScene *assimpScene = importer.ReadFile(filename.data(), (uint)aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
																		aiProcess_GenUVCoords | aiProcess_FindInstances | aiProcess_RemoveRedundantMaterials);

	if (!assimpScene)
	{
		const std::string error = "Could not load file: " + std::string(filename) + ", error: " + std::string(importer.GetErrorString());
		WARNING(error.c_str());
		throw LoadException(error);
	}

	m_IsAnimated = assimpScene->HasAnimations();
	std::vector<vec2> textureCoords;

	std::vector<uint> matMapping(assimpScene->mNumMaterials);
	if (material < 0)
	{
		for (uint i = 0; i < assimpScene->mNumMaterials; i++)
			matMapping.at(i) = matList->add(assimpScene->mMaterials[i], directory);
	}
	else
	{
		for (uint i = 0; i < assimpScene->mNumMaterials; i++)
			matMapping.at(i) = material;
	}

	bool hasSkin = false;
	rfw::MeshSkin skin = {};
	skin.jointMatrices.push_back(glm::mat4(1.0f));

	// Create our own scene graph from Assimp node graph, store every node's transformation along the way
	// Assimp supports trivial instancing support, thus we need to traverse the scene graph to determine
	// how many times each mesh occurs
	traverseNode(assimpScene->mRootNode, &m_NodeNameMapping);
	scene.rootNodes.push_back(0);

	for (const auto n : m_NodesWithMeshes)
	{
		const auto &node = scene.nodes.at(n);

		for (const auto meshID : scene.nodes.at(n).meshIDs)
		{
			const aiMesh *curMesh = assimpScene->mMeshes[m_AssimpMeshMapping.at(meshID)];

			const auto vertexOffset = scene.baseVertices.size();
			const auto faceOffset = scene.indices.size();

			rfw::SceneMesh &m = scene.meshes.at(meshID);
			m.object = &scene;

			// Store location & range info
			m.vertexOffset = vertexOffset;
			m.vertexCount = curMesh->mNumVertices;
			m.faceOffset = faceOffset;
			m.faceCount = curMesh->mNumFaces;
			m.matID = matMapping[curMesh->mMaterialIndex];

			// Allocate storage
			scene.baseVertices.reserve(scene.baseVertices.size() + curMesh->mNumVertices);
			scene.baseNormals.reserve(scene.baseNormals.size() + curMesh->mNumVertices);
			textureCoords.reserve(textureCoords.size() + curMesh->mNumVertices);
			scene.materialIndices.reserve(scene.triangles.size() + curMesh->mNumFaces);
			scene.triangles.resize(scene.triangles.size() + curMesh->mNumFaces);

			// Store vertices
			for (uint v = 0; v < curMesh->mNumVertices; v++)
			{
				const vec3 vertex = glm::make_vec3(&curMesh->mVertices[v].x);
				const vec3 normal = glm::make_vec3(&curMesh->mNormals[v].x);

				scene.baseVertices.push_back(vec4(vertex, 1.0f));
				scene.baseNormals.push_back(normal);

				if (curMesh->HasTextureCoords(0))
					textureCoords.emplace_back(curMesh->mTextureCoords[0][v].x, curMesh->mTextureCoords[0][v].y);
				else
					textureCoords.emplace_back(0.0f, 0.0f);
			}

			// Store indices
			for (uint f = 0; f < curMesh->mNumFaces; f++)
			{
				const aiFace &face = curMesh->mFaces[f];
				assert(face.mNumIndices == 3);

				const auto f0 = face.mIndices[0];
				const auto f1 = face.mIndices[1];
				const auto f2 = face.mIndices[2];

				assert(f0 < curMesh->mNumVertices);
				assert(f1 < curMesh->mNumVertices);
				assert(f2 < curMesh->mNumVertices);

				scene.indices.push_back(uvec3(f0, f1, f2) + uint(vertexOffset));
				scene.materialIndices.push_back(matMapping[curMesh->mMaterialIndex]);
			}

			if (curMesh->HasBones())
			{
				hasSkin = true;
				m.weights.resize(m.vertexCount, vec4(0.0f));
				m.joints.resize(m.vertexCount, uvec4(0));

				for (int boneIdx = 0; boneIdx < curMesh->mNumBones && boneIdx < 4; boneIdx++)
				{
					const aiBone *bone = curMesh->mBones[boneIdx];

					const int jointIdx = skin.jointMatrices.size();
					glm::mat4 matrix;
					memcpy(value_ptr(matrix), &bone->mOffsetMatrix[0][0], sizeof(float) * 16);
					skin.jointMatrices.push_back(matrix);

					for (int i = 0; i < curMesh->mNumVertices; i++)
						m.joints.at(i)[boneIdx] = jointIdx;

					for (int i = 0; i < bone->mNumWeights; i++)
					{
						const auto vId = bone->mWeights[i].mVertexId;
						m.weights.at(vId)[boneIdx] = bone->mWeights[i].mWeight;
					}
				}
			}
		}
	}

	if (hasSkin)
		scene.skins.push_back(skin);

	for (int i = 0; i < scene.nodes.size(); i++)
	{
		auto &node = scene.nodes.at(i);
		node.skinIDs.resize(node.meshIDs.size(), -1);

		if (node.meshIDs.empty())
			continue;

		for (int j = 0; j < node.meshIDs.size(); j++)
		{
			const auto mesh = assimpScene->mMeshes[node.meshIDs.at(j)];
			if (mesh->HasBones())
				node.skinIDs.at(j) = 0;
		}
	}

	if (m_IsAnimated)
	{
		for (int i = 0; i < assimpScene->mNumAnimations; i++)
		{
			rfw::SceneAnimation sceneAnimation = {};
			sceneAnimation.object = &scene;

			const aiAnimation *anim = assimpScene->mAnimations[i];
			for (int j = 0; j < anim->mNumChannels; j++)
			{
				const aiNodeAnim *nodeAnim = anim->mChannels[j];

				if (nodeAnim->mNumPositionKeys > 0)
				{
					rfw::SceneAnimation::Channel channel = {};
					channel.nodeIdx = m_NodeNameMapping[nodeAnim->mNodeName.C_Str()];
					channel.target = SceneAnimation::Channel::TRANSLATION;
					channel.k = nodeAnim->mNumPositionKeys;

					rfw::SceneAnimation::Sampler sampler = {};
					sampler.method = SceneAnimation::Sampler::LINEAR;
					sampler.t.reserve(nodeAnim->mNumPositionKeys);
					sampler.vec3_key.reserve(nodeAnim->mNumPositionKeys);
					for (int k = 0; k < nodeAnim->mNumPositionKeys; k++)
					{
						const aiVectorKey &key = nodeAnim->mPositionKeys[k];
						sampler.t.push_back(key.mTime);
						sampler.vec3_key.push_back(vec3(key.mValue.x, key.mValue.y, key.mValue.z));
					}

					channel.samplerIdx = sceneAnimation.samplers.size();
					sceneAnimation.samplers.emplace_back(sampler);
					sceneAnimation.channels.push_back(channel);
				}

				if (nodeAnim->mNumRotationKeys > 0)
				{
					rfw::SceneAnimation::Channel channel = {};
					channel.nodeIdx = m_NodeNameMapping[nodeAnim->mNodeName.C_Str()];
					channel.target = SceneAnimation::Channel::ROTATION;
					channel.k = nodeAnim->mNumRotationKeys;

					rfw::SceneAnimation::Sampler sampler = {};
					sampler.method = SceneAnimation::Sampler::LINEAR;
					sampler.t.reserve(nodeAnim->mNumRotationKeys);
					sampler.vec3_key.reserve(nodeAnim->mNumRotationKeys);
					for (int k = 0; k < nodeAnim->mNumRotationKeys; k++)
					{
						const aiQuatKey &key = nodeAnim->mRotationKeys[k];
						sampler.t.push_back(key.mTime);
						sampler.quat_key.push_back(glm::quat(key.mValue.w, key.mValue.x, key.mValue.y, key.mValue.z));
					}

					channel.samplerIdx = sceneAnimation.samplers.size();
					sceneAnimation.samplers.emplace_back(sampler);
					sceneAnimation.channels.push_back(channel);
				}

				if (nodeAnim->mNumScalingKeys > 0)
				{
					rfw::SceneAnimation::Channel channel = {};
					channel.nodeIdx = m_NodeNameMapping[nodeAnim->mNodeName.C_Str()];
					channel.target = SceneAnimation::Channel::SCALE;
					channel.k = nodeAnim->mNumScalingKeys;

					rfw::SceneAnimation::Sampler sampler = {};
					sampler.method = SceneAnimation::Sampler::LINEAR;
					sampler.t.reserve(nodeAnim->mNumScalingKeys);
					sampler.vec3_key.reserve(nodeAnim->mNumScalingKeys);
					for (int k = 0; k < nodeAnim->mNumScalingKeys; k++)
					{
						const aiVectorKey &key = nodeAnim->mScalingKeys[k];
						sampler.t.push_back(key.mTime);
						sampler.vec3_key.push_back(vec3(key.mValue.x, key.mValue.y, key.mValue.z));
					}

					channel.samplerIdx = sceneAnimation.samplers.size();
					sceneAnimation.samplers.emplace_back(sampler);
					sceneAnimation.channels.push_back(channel);
				}
			}

			sceneAnimation.ticksPerSecond = anim->mTicksPerSecond;
			sceneAnimation.duration = anim->mDuration;
			scene.animations.push_back(sceneAnimation);
		}
	}

	// Transform meshes according to node transformations in scene graph
	scene.transformTo(0.0f);

	// Update triangle data that only has to be calculated once
	scene.updateTriangles(matList, textureCoords);
	scene.updateTriangles();

	utils::logger::log("Loaded \"%s\" with triangle count: %zu", filename.data(), scene.triangles.size());
}

void AssimpObject::transformTo(const float timeInSeconds) { scene.transformTo(timeInSeconds); }

size_t rfw::AssimpObject::traverseNode(const aiNode *node, std::map<std::string, uint> *nodeNameMapping)
{
	// Allocate node
	const size_t currentNodeIndex = scene.nodes.size();
	scene.nodes.push_back(rfw::SceneNode(&scene, node->mName.C_Str(), {}));

	// Add index to current node to name mapping
	(*nodeNameMapping)[std::string(node->mName.C_Str())] = static_cast<uint>(currentNodeIndex);

	if (node->mNumMeshes > 0)
	{
		m_NodesWithMeshes.push_back(currentNodeIndex);
		scene.nodes.at(currentNodeIndex).meshIDs.resize(node->mNumMeshes);
		scene.nodes.at(currentNodeIndex).skinIDs.resize(node->mNumMeshes, -1);

		for (uint i = 0; i < node->mNumMeshes; i++)
		{
			const auto meshIdx = scene.meshes.size();
			scene.meshes.emplace_back();
			scene.meshes.at(meshIdx).object = &scene;

			m_AssimpMeshMapping.push_back(node->mMeshes[i]);
			scene.nodes.at(currentNodeIndex).meshIDs.at(i) = meshIdx;
		}
	}

	// Assimp provides matrices in row-major form, convert to glm-compatible column major
	glm::mat4 matrix;
	memcpy(value_ptr(matrix), &node->mTransformation[0][0], 16 * sizeof(float));
	scene.nodes.at(currentNodeIndex).matrix = glm::rowMajor4(matrix);
	scene.nodes.at(currentNodeIndex).calculateTransform();
	scene.nodes.at(currentNodeIndex).childIndices.reserve(node->mNumChildren);

	// Iterate and initialize children
	std::vector<int> children;
	for (uint i = 0; i < node->mNumChildren; i++)
	{
		const aiNode *child = node->mChildren[i];
		children.push_back(static_cast<int>(traverseNode(child, nodeNameMapping)));
	}

	scene.nodes.at(currentNodeIndex).childIndices = children;
	return currentNodeIndex;
}

uint AssimpObject::getAnimationCount() const { return static_cast<uint>(scene.animations.size()); }

void AssimpObject::setAnimation(uint index) { m_CurrentAnimation = clamp(index, 0u, static_cast<uint>(scene.animations.size())); }

uint AssimpObject::getMaterialForPrim(uint primitiveIdx) const { return scene.materialIndices.at(primitiveIdx); }

void AssimpObject::setCurrentAnimation(uint index)
{
	assert(index < getAnimationCount());
	m_CurrentAnimation = index;
}

std::vector<uint> AssimpObject::getLightIndices(const std::vector<bool> &matLightFlags) const
{
	std::vector<uint> indices;
	for (const auto &mesh : scene.meshes)
	{
		const size_t offset = indices.size();
		if (matLightFlags.at(mesh.matID))
		{
			const auto s = mesh.vertexCount / 3;
			const auto o = mesh.vertexOffset / 3;
			indices.resize(offset + s);
			for (uint i = 0; i < s; i++)
				indices.at(offset + i) = o + i;
		}
	}

	return indices;
}