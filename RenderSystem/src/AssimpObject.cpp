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

struct MatrixArray
{
	float data[16];
};

inline MatrixArray columnToRow(const float *columnMatrix)
{
	MatrixArray rowMatrix;
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			rowMatrix.data[j * 4 + i] = rowMatrix.data[i * 4 + j];
	}

	return rowMatrix;
}

inline MatrixArray rowToColumn(const float *rowMatrix)
{
	MatrixArray columnMatrix{};
	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
			columnMatrix.data[i * 4 + j] = rowMatrix[j * 4 + i];
	}

	return columnMatrix;
}

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

void glm_to_ai(const glm::mat4 &from, aiMatrix4x4 &to)
{
	glm::mat4 newMat = glm::transpose(from);
	for (int i = 0; i < 4; ++i)
	{
		for (int j = 0; j < 4; ++j)
		{
			to[i][j] = newMat[i][j];
		}
	}
}

AssimpObject::AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const mat4 &matrix,
						   bool normalize, int material)
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

	const aiScene *scene = importer.ReadFile(
		filename.data(), (uint)aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
							 aiProcess_GenUVCoords | aiProcess_FindInstances | aiProcess_RemoveRedundantMaterials);

	if (!scene)
	{
		const std::string error =
			"Could not load file: " + std::string(filename) + ", error: " + std::string(importer.GetErrorString());
		WARNING(error.c_str());
		throw LoadException(error);
	}

	m_IsAnimated = scene->HasAnimations();

	// Create our own scene graph from assimp node graph, store every node's transformation along the way
	traverseNode(scene->mRootNode, -1, &m_SceneGraph, &m_BaseNodeTransformations, &m_NodeNameMapping);
	calculateMatrices(m_SceneGraph.at(0), m_SceneGraph, m_BaseNodeTransformations);

	if (m_IsAnimated)
	{
		m_Animations.resize(scene->mNumAnimations);
		for (uint i = 0; i < scene->mNumAnimations; i++)
		{
			const aiAnimation *anim = scene->mAnimations[i];
			MeshAnimation &animation = m_Animations.at(i);
			animation.name = std::string(anim->mName.C_Str());
			animation.duration = anim->mDuration;
			animation.ticksPerSecond = anim->mTicksPerSecond;

			animation.channels.resize(anim->mNumChannels);
			for (uint c = 0; c < anim->mNumChannels; c++)
			{
				const aiNodeAnim *nodeAnim = anim->mChannels[c];
				AnimationChannel &channel = animation.channels.at(c);
				channel.nodeName = std::string(nodeAnim->mNodeName.C_Str());
				channel.nodeIndex = m_NodeNameMapping[channel.nodeName];
				channel.preState = nodeAnim->mPreState;
				channel.postState = nodeAnim->mPostState;

				channel.positionKeys.resize(nodeAnim->mNumPositionKeys);
				channel.rotationKeys.resize(nodeAnim->mNumRotationKeys);
				channel.scalingKeys.resize(nodeAnim->mNumScalingKeys);

				for (uint j = 0; j < nodeAnim->mNumPositionKeys; j++)
					channel.positionKeys.at(j) = {nodeAnim->mPositionKeys[j].mTime,
												  vec3(nodeAnim->mPositionKeys[j].mValue.x,
													   nodeAnim->mPositionKeys[j].mValue.y,
													   nodeAnim->mPositionKeys[j].mValue.z)};

				for (uint j = 0; j < nodeAnim->mNumRotationKeys; j++)
				{
					const aiQuatKey &key = nodeAnim->mRotationKeys[j];
					// const glm::quat q = glm::quat(key.mValue.w, key.mValue.x, key.mValue.y, key.mValue.z);
					channel.rotationKeys.at(j) = {key.mTime, key.mValue};
				}

				for (uint j = 0; j < nodeAnim->mNumScalingKeys; j++)
					channel.scalingKeys.at(j) = {nodeAnim->mScalingKeys[j].mTime,
												 vec3(nodeAnim->mScalingKeys[j].mValue.x,
													  nodeAnim->mScalingKeys[j].mValue.y,
													  nodeAnim->mScalingKeys[j].mValue.z)};
			}

			animation.meshChannels.resize(anim->mNumMeshChannels);
			for (uint c = 0; c < anim->mNumMeshChannels; c++)
			{
				const aiMeshAnim *nodeAnim = anim->mMeshChannels[c];
				MeshAnimationChannel &channel = animation.meshChannels.at(c);
				channel.name = std::string(nodeAnim->mName.C_Str());

				// Look up mesh index to prevent having to search for it
				for (uint m = 0; m < scene->mNumMeshes; m++)
				{
					const std::string name = std::string(scene->mMeshes[m]->mName.C_Str());
					if (channel.name == name)
					{
						channel.meshIndex = m;
						break;
					}
				}

				channel.keys.resize(nodeAnim->mNumKeys);
				for (uint k = 0; k < nodeAnim->mNumKeys; k++)
				{
					const aiMeshKey &key = nodeAnim->mKeys[k];
					channel.keys.at(k).time = float(key.mTime);
					channel.keys.at(k).value = key.mValue;
				}
			}

			animation.meshChannels.resize(anim->mNumMorphMeshChannels);
			for (uint c = 0; c < anim->mNumMorphMeshChannels; c++)
			{
				const aiMeshMorphAnim *nodeAnim = anim->mMorphMeshChannels[c];
				MeshMorphAnimation &channel = animation.morphChannels.at(c);
				channel.name = std::string(nodeAnim->mName.C_Str());

				// Look up mesh index to prevent having to search for it
				for (uint m = 0; m < scene->mNumMeshes; m++)
				{
					const std::string name = std::string(scene->mMeshes[m]->mName.C_Str());
					if (channel.name == name)
					{
						channel.meshIndex = m;
						break;
					}
				}

				channel.keys.resize(nodeAnim->mNumKeys);
				for (uint k = 0; k < nodeAnim->mNumKeys; k++)
				{
					const aiMeshMorphKey aiKey = nodeAnim->mKeys[k];
					channel.keys.at(k).time = float(aiKey.mTime);
					channel.keys.at(k).values.resize(aiKey.mNumValuesAndWeights);
					channel.keys.at(k).weights.resize(aiKey.mNumValuesAndWeights);
					for (uint v = 0; v < aiKey.mNumValuesAndWeights; v++)
					{
						channel.keys.at(k).values.at(v) = aiKey.mValues[v];
						channel.keys.at(k).weights.at(v) = float(aiKey.mWeights[v]);
					}
				}
			}
		}
	}

	std::vector<uint> matMapping(scene->mNumMaterials);

	if (material < 0)
	{
		for (uint i = 0; i < scene->mNumMaterials; i++)
			matMapping.at(i) = matList->add(scene->mMaterials[i], directory);
	}
	else
	{
		for (uint i = 0; i < scene->mNumMaterials; i++)
			matMapping.at(i) = material;
	}

	size_t meshCount = 0;
	size_t vertexCount = 0;
	size_t faceCount = 0;
	for (const auto &nodeIdx : m_NodesWithMeshes)
	{
		const auto &node = m_SceneGraph.at(nodeIdx);
		meshCount += node.meshes.size();
		for (const auto m : node.meshes)
		{
			vertexCount += scene->mMeshes[m]->mNumVertices;
			faceCount += scene->mMeshes[m]->mNumFaces;
		}
	}

	m_Meshes.resize(meshCount);
	m_MeshMapping.resize(meshCount);

	m_CurrentVertices.resize(vertexCount);
	m_CurrentNormals.resize(vertexCount);

	m_BaseVertices.resize(vertexCount);
	m_BaseNormals.resize(vertexCount);
	m_BaseTexCoords.resize(vertexCount);

	m_Indices.resize(faceCount);
	m_MaterialIndices.resize(faceCount);

	uint vertexOffset = 0;
	uint faceOffset = 0;
	uint meshIndex = 0;
	for (const auto &nodeIdx : m_NodesWithMeshes)
	{
		const auto &node = m_SceneGraph.at(nodeIdx);
		for (uint i = 0; i < node.meshes.size(); i++)
		{
			const auto meshIdx = node.meshes.at(i);
			m_MeshMapping.at(meshIdx).push_back(meshIndex);

			MeshInfo &m = m_Meshes.at(meshIndex);
			const aiMesh *curMesh = scene->mMeshes[meshIdx];
			const uint matIndex = matMapping[curMesh->mMaterialIndex];

			for (uint v = 0; v < curMesh->mNumVertices; v++)
			{
				const auto vIdx = vertexOffset + v;

				vec4 vertex = vec4(glm::make_vec3(&curMesh->mVertices[v].x), 1.0f);
				vec3 normal = glm::make_vec3(&curMesh->mNormals[v].x);

				if (hasTransform)
				{
					vertex = matrix * vertex;
					normal = matrix3x3 * normal;
				}

				m_BaseVertices.at(vIdx) = vertex;
				m_BaseNormals.at(vIdx) = normal;
				if (curMesh->mTextureCoords[0])
					m_BaseTexCoords.at(vIdx) = vec2(curMesh->mTextureCoords[0][v].x, curMesh->mTextureCoords[0][v].y);
				else
					m_BaseTexCoords.at(vIdx) = vec2(0.0f);
			}

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

				const vec3 v0 = glm::make_vec3(&curMesh->mVertices[f0].x);
				const vec3 v1 = glm::make_vec3(&curMesh->mVertices[f1].x);
				const vec3 v2 = glm::make_vec3(&curMesh->mVertices[f2].x);

				const vec3 vn0 = glm::make_vec3(&curMesh->mNormals[f0].x);
				const vec3 vn1 = glm::make_vec3(&curMesh->mNormals[f1].x);
				const vec3 vn2 = glm::make_vec3(&curMesh->mNormals[f2].x);

				vec3 N = glm::normalize(glm::cross(v1 - v0, v2 - v0));
				if (dot(N, vn0) < 0 && dot(N, vn1) < 0 && dot(N, vn2) < 0)
					N *= -1.0f; // flip if not consistent with vertex normals

				const auto fIdx = faceOffset + f;
				m_MaterialIndices.at(fIdx) = matIndex;
				m_Indices.at(fIdx) = uvec3(f0, f1, f2) + vertexOffset;
			}

			if (curMesh->HasBones())
			{
				m.bones.resize(curMesh->mNumBones);

				for (uint boneIdx = 0; boneIdx < curMesh->mNumBones; boneIdx++)
				{
					const aiBone *aiBone = curMesh->mBones[boneIdx];
					// Store bone name
					MeshBone &bone = m.bones.at(boneIdx);
					bone.name = std::string(aiBone->mName.C_Str());

					const aiNode *node = scene->mRootNode->FindNode(aiBone->mName);
					bone.nodeIndex = m_NodeNameMapping[std::string(node->mName.C_Str())];
					bone.nodeName = std::string(node->mName.C_Str());

					// Store bone weights
					bone.boneWeights.resize(aiBone->mNumWeights);
					for (uint w = 0; w < aiBone->mNumWeights; w++)
					{
						MeshBoneWeight &weight = bone.boneWeights.at(w);
						weight.vertexId = aiBone->mWeights[w].mVertexId;
						weight.weight = aiBone->mWeights[w].mWeight;
					}

					bone.offsetMatrix =
						glm::make_mat4(rowToColumn(const_cast<float *>(&aiBone->mOffsetMatrix.a1)).data);
					bone.aiOffsetMatrix = aiBone->mOffsetMatrix;
				}
			}

			m.nodeIndex = static_cast<uint>(nodeIdx);
			m.vertexOffset = vertexOffset;
			m.vertexCount = curMesh->mNumVertices;
			m.faceOffset = faceOffset;
			m.faceCount = curMesh->mNumFaces;
			m.materialIdx = matMapping.at(curMesh->mMaterialIndex);

			vertexOffset = vertexOffset + curMesh->mNumVertices;
			faceOffset = faceOffset + curMesh->mNumFaces;
			meshIndex++;
		}
	}

	m_CurrentVertices = m_BaseVertices;
	m_CurrentNormals = m_BaseNormals;

	// Transform meshes according to node transformations in scene graph
	transformTo(0.0f);

	// Update triangle data that only has to be calculated once
	for (auto &tri : m_Triangles)
	{
		HostMaterial mat = matList->get(tri.material);

		int texID = mat.map[0].textureID;
		if (texID > -1)
		{
			const Texture &texture = matList->getTextures().at(texID);

			const float Ta = static_cast<float>(texture.width * texture.height) *
							 abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
			const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			tri.LOD = 0.5f * log2f(Ta / Pa);
		}
	}

	char buffer[1024];
	sprintf(buffer, "Loaded \"%s\" with triangle count: %zu", filename.data(), m_Triangles.size());
	DEBUG(buffer);
}

void AssimpObject::transformTo(const float timeInSeconds)
{
	if (!m_IsAnimated && m_HasUpdated)
		return;
	m_HasUpdated = true;

	// Start of with regular node transformations
	std::vector<aiMatrix4x4> trsNodeTransformations = m_BaseNodeTransformations;

	// Start of with clean base data
	m_CurrentVertices.resize(m_BaseVertices.size());
	m_CurrentNormals.resize(m_BaseNormals.size());
	memset(m_CurrentVertices.data(), 0, m_CurrentVertices.size() * sizeof(vec4));
	memset(m_CurrentNormals.data(), 0, m_CurrentNormals.size() * sizeof(vec3));

	if (m_IsAnimated)
	{
		const float timeInTicks = timeInSeconds * static_cast<float>(m_Animations[0].ticksPerSecond);
		const float animationTime = std::fmod(timeInTicks, static_cast<float>(m_Animations[0].duration));

#if ANIMATION_ENABLED
		// TODO: Implement mesh animations and mesh morph animations
		// Replace node transformations with animation transformations
		for (const auto &animation : m_Animations)
		{
			for (const auto &channel : animation.channels)
			{
				auto &location = trsNodeTransformations.at(channel.nodeIndex);
				glm_to_ai(channel.getInterpolatedTRS(animationTime), location);
			}

			for (const auto &channel : animation.meshChannels)
			{
				// TODO
			}

			for (const auto &channel : animation.morphChannels)
			{
				// TODO
			}
		}
		// Pre calculate top-down matrices for every node
		calculateMatrices(m_SceneGraph.at(0), m_SceneGraph, trsNodeTransformations);
#endif
	}

	for (const auto &mesh : m_Meshes)
	{
		// Set current data for this mesh to base data
		if (mesh.bones.empty())
		{
			/*
			 * Unfortunately, this part of the loop cannot be done on initialization.
			 * Meshes with animations can have sub-meshes that are placed in a node with supposedly no animations.
			 * Nodes can have children though and thus a node can be influenced by an upper node with animations.
			 */
			const aiMatrix4x4 &transform = m_aiPrecalcNodeTransformations.at(mesh.nodeIndex);
			auto transform3x3 = aiMatrix3x3(transform);
			for (uint i = 0; i < mesh.vertexCount; i++)
			{
				const auto vIdx = mesh.vertexOffset + i;
				const auto baseVertex = m_BaseVertices.at(vIdx);
				const auto baseNormal = m_BaseNormals.at(vIdx);

				aiVector3D pos = aiVector3D(baseVertex.x, baseVertex.y, baseVertex.z);
				aiVector3D normal = aiVector3D(baseNormal.x, baseNormal.y, baseNormal.z);

				aiTransformVecByMatrix4(&pos, &transform);
				aiTransformVecByMatrix3(&normal, &transform3x3);

				const auto index = static_cast<size_t>(mesh.vertexOffset + i);

				m_CurrentVertices.at(index) = glm::vec4(pos.x, pos.y, pos.z, 1.0f);
				m_CurrentNormals.at(index) = glm::vec3(normal.x, normal.y, normal.z);
			}
			continue;
		}

		for (const auto &bone : mesh.bones)
		{
			aiMatrix4x4 nodeTransform = m_aiPrecalcNodeTransformations.at(mesh.nodeIndex);
			aiMatrix4x4 skin4x4 = m_aiPrecalcNodeTransformations.at(bone.nodeIndex);
			aiMultiplyMatrix4(&skin4x4, &bone.aiOffsetMatrix);

			auto skin3x3 = aiMatrix3x3(skin4x4);
			skin3x3.Inverse();

			for (const auto &weight : bone.boneWeights)
			{
				const uint vIdx = mesh.vertexOffset + weight.vertexId;
				aiVector3D pos =
					aiVector3D(m_BaseVertices.at(vIdx).x, m_BaseVertices.at(vIdx).y, m_BaseVertices.at(vIdx).z);
				aiTransformVecByMatrix4(&pos, &skin4x4);
				aiVector3D n = aiVector3D(m_BaseNormals.at(vIdx).x, m_BaseNormals.at(vIdx).y, m_BaseNormals.at(vIdx).z);
				aiTransformVecByMatrix3(&n, &skin3x3);

				const vec3 vertex = glm::vec3(pos.x, pos.y, pos.z) * weight.weight;
				const vec3 normal = glm::vec3(n.x, n.y, n.z) * weight.weight;

				m_CurrentVertices.at(vIdx) += vec4(vertex, 0.0f);
				m_CurrentNormals.at(vIdx) += normal;
			}
		}
	}

	updateTriangles();

	// In case this object has no animations, we can clear all data after loading and transforming all data
	if (m_HasUpdated && !m_IsAnimated)
	{
		m_BaseNormals.clear();
		m_BaseNodeTransformations.clear();
		m_BaseTexCoords.clear();
		m_BaseVertices.clear();
	}
}

void AssimpObject::updateTriangles()
{
	m_Triangles.resize(m_Indices.size());
	for (size_t i = 0, s = m_Triangles.size(); i < s; i++)
	{
		Triangle &tri = m_Triangles.at(i);
		const glm::uvec3 &indices = m_Indices.at(i);
		const vec3 &v0 = m_CurrentVertices.at(indices.x);
		const vec3 &v1 = m_CurrentVertices.at(indices.y);
		const vec3 &v2 = m_CurrentVertices.at(indices.z);

		const vec3 &n0 = m_CurrentNormals.at(indices.x);
		const vec3 &n1 = m_CurrentNormals.at(indices.y);
		const vec3 &n2 = m_CurrentNormals.at(indices.z);

		tri.vertex0 = v0;
		tri.vertex1 = v1;
		tri.vertex2 = v2;

		tri.u0 = m_BaseTexCoords.at(indices.x).x;
		tri.v0 = m_BaseTexCoords.at(indices.x).y;

		tri.u1 = m_BaseTexCoords.at(indices.y).x;
		tri.v1 = m_BaseTexCoords.at(indices.y).y;

		tri.u2 = m_BaseTexCoords.at(indices.z).x;
		tri.v2 = m_BaseTexCoords.at(indices.z).y;

		vec3 N = normalize(cross(v1 - v0, v2 - v0));

		if (dot(N, n0) < 0.0f && dot(N, n1) < 0.0f && dot(N, n1) < 0.0f)
			N *= -1.0f; // flip if not consistent with vertex normals

		tri.Nx = N.x;
		tri.Ny = N.y;
		tri.Nz = N.z;

		tri.vN0 = n0;
		tri.vN1 = n1;
		tri.vN2 = n2;

		tri.material = m_MaterialIndices.at(i);
	}
}

size_t rfw::AssimpObject::traverseNode(const aiNode *node, int parentIdx, std::vector<AssimpNode> *storage,
									   std::vector<aiMatrix4x4> *matrixStorage,
									   std::map<std::string, uint> *nodeNameMapping)
{
	// Get current index
	const size_t currentNodeIndex = storage->size();
	// Add index to current node to name mapping
	(*nodeNameMapping)[std::string(node->mName.C_Str())] = static_cast<uint>(currentNodeIndex);

	// Initialize node data
	AssimpNode n;
	n.parent = parentIdx;
	n.children.resize(node->mNumChildren);

	if (node->mNumMeshes > 0)
	{
		n.meshes.resize(node->mNumMeshes);
		for (uint i = 0; i < node->mNumMeshes; i++)
		{
			const auto meshIdx = node->mMeshes[i];
			n.meshes.at(i) = meshIdx;
		}

		m_NodesWithMeshes.push_back(currentNodeIndex);
	}

	matrixStorage->push_back(node->mTransformation);
	storage->push_back(n);

	// Iterate and initialize children
	const int currentIdx = static_cast<int>(currentNodeIndex);
	for (uint i = 0; i < node->mNumChildren; i++)
	{
		const aiNode *child = node->mChildren[i];
		const size_t childIdx = traverseNode(child, currentIdx, storage, matrixStorage, nodeNameMapping);
		storage->at(currentNodeIndex).children.at(i) = static_cast<uint>(childIdx);
	}

	return currentNodeIndex;
}

// Iterate over every node and its children, calculating every node's matrix
void rfw::AssimpObject::calculateMatrices(const AssimpNode &node, const std::vector<AssimpNode> &nodes,
										  const std::vector<aiMatrix4x4> &perNodeMatrices)
{
	m_aiPrecalcNodeTransformations.resize(nodes.size());

	const auto traverseTillParent = [&perNodeMatrices, &nodes](int currentIndex) -> aiMatrix4x4 {
		std::vector<const aiMatrix4x4 *> matrices;
		while (currentIndex != -1)
		{
			matrices.push_back(&perNodeMatrices.at(currentIndex));
			currentIndex = nodes.at(currentIndex).parent;
		}

		aiMatrix4x4 result;
		aiIdentityMatrix4(&result);
		for (int i = static_cast<int>(matrices.size()) - 1; i >= 0; i--)
			aiMultiplyMatrix4(&result, matrices.at(i));

		return result;
	};

	for (size_t i = 0, s = nodes.size(); i < s; i++)
		m_aiPrecalcNodeTransformations.at(i) = traverseTillParent(static_cast<int>(i));
}
uint AssimpObject::getAnimationCount() const { return static_cast<uint>(m_Animations.size()); }

void AssimpObject::setAnimation(uint index)
{
	m_CurrentAnimation = clamp(index, 0u, static_cast<uint>(m_Animations.size()));
}

uint AssimpObject::getMaterialForPrim(uint primitiveIdx) const
{
	for (const auto &mesh : m_Meshes)
	{
		if (mesh.faceOffset < primitiveIdx && (mesh.faceOffset + mesh.faceCount) > primitiveIdx)
			return mesh.materialIdx;
	}

	return 0;
}

void AssimpObject::setCurrentAnimation(uint index)
{
	assert(index < getAnimationCount());
	m_CurrentAnimation = index;
}

std::vector<uint> AssimpObject::getLightIndices(const std::vector<bool> &matLightFlags) const
{
	std::vector<uint> indices;
	for (const auto &mesh : m_Meshes)
	{
		const size_t offset = indices.size();
		if (matLightFlags.at(mesh.materialIdx))
		{
			indices.resize(offset + mesh.faceCount);
			for (uint i = 0; i < mesh.faceCount; i++)
				indices.at(offset + i) = mesh.faceOffset + i;
		}
	}

	return indices;
}

glm::mat4 rfw::AssimpObject::AnimationChannel::getInterpolatedTRS(float time) const
{
	vec3 position = vec3(0.0f);
	if (!positionKeys.empty())
	{

		size_t posKeyIndex = 0, posKeyIndexNext = 0;
		for (size_t i = 0, s = positionKeys.size() - 1; i < s; i++)
		{
			if (time < positionKeys.at(i).time)
			{
				posKeyIndex = i, posKeyIndexNext = i + 1;
				break;
			}
		}

		assert(posKeyIndexNext < positionKeys.size());
		const float posDeltaTime = positionKeys.at(posKeyIndexNext).time - positionKeys.at(posKeyIndex).time;
		const float posFactor = (time - positionKeys.at(posKeyIndex).time) / posDeltaTime;
		const vec3 &startPos = positionKeys.at(posKeyIndex).value;
		const vec3 &endPos = positionKeys.at(posKeyIndexNext).value;
		position = startPos + posFactor * (endPos - startPos);
	}

	glm::quat q = glm::identity<glm::quat>();
	if (!rotationKeys.empty())
	{

		size_t rotKeyIndex = 0, rotKeyIndexNext = 0;
		for (size_t i = 0, s = rotationKeys.size() - 1; i < s; i++)
		{
			if (time < rotationKeys.at(i).time)
			{
				rotKeyIndex = i, rotKeyIndexNext = i + 1;
				break;
			}
		}
		assert(rotKeyIndexNext < rotationKeys.size());
		const float rotDeltaTime = rotationKeys.at(rotKeyIndexNext).time - rotationKeys.at(rotKeyIndex).time;
		const float rotFactor = (time - rotationKeys.at(rotKeyIndex).time) / rotDeltaTime;
		const auto &startQuat = rotationKeys.at(rotKeyIndex).value;
		const auto &endQuat = rotationKeys.at(rotKeyIndexNext).value;
		q = glm::mix(startQuat, endQuat, rotFactor);
	}

	vec3 scaling = vec3(1.0f);
	if (!scalingKeys.empty())
	{
		size_t scaleKeyIndex = 0, scaleKeyIndexNext = 0;
		for (size_t i = 0, s = scalingKeys.size() - 1; i < s; i++)
		{
			if (time < scalingKeys.at(i).time)
			{
				scaleKeyIndex = i, scaleKeyIndexNext = i + 1;
				break;
			}
		}
		assert(scaleKeyIndexNext < scalingKeys.size());
		const float scaleDeltaTime = scalingKeys.at(scaleKeyIndexNext).time - scalingKeys.at(scaleKeyIndex).time;
		const float scaleFactor = (time - scalingKeys.at(scaleKeyIndex).time) / scaleDeltaTime;
		const vec3 &startScale = scalingKeys.at(scaleKeyIndex).value;
		const vec3 &endScale = scalingKeys.at(scaleKeyIndexNext).value;
		scaling = startScale + scaleFactor * (endScale - startScale);
	}

	const glm::mat4 T = glm::translate(glm::mat4(1.0f), position);
	const glm::mat4 R = glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));
	const glm::mat4 S = glm::scale(glm::mat4(1.0f), scaling);

	return T * R * S;
}