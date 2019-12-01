#define GLM_FORCE_SIMD_AVX2
#include "AssimpObject.h"

#include <MathIncludes.h>

#include <array>
#include <cmath>
#include <set>

#include <glm/gtx/matrix_major_storage.hpp>

#include <assimp/cimport.h>
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

struct NodeWithIndex
{
	std::string name;
	const aiNode *node{};
	int index{};
	mat4 transform{};
	std::vector<int> childIndices;
	std::vector<int> meshIDs;
};

int calculateNodeCount(const aiNode *node)
{
	int count = 1;
	for (int i = 0; i < node->mNumChildren; i++)
		count += calculateNodeCount(node->mChildren[i]);
	return count;
}

int traverseNodeTree(NodeWithIndex* storage, const aiNode *node, int &counter)
{
	const int index = counter++;

	storage[index].name = node->mName.C_Str();
	storage[index].node = node;
	storage[index].index = index;

	mat4 T;
	memcpy(value_ptr(T), &node->mTransformation, 16 * sizeof(float));
	T = rowMajor4(T);
	storage[index].transform = T;

	for (int i = 0; i < node->mNumMeshes; i++)
		storage[index].meshIDs.push_back(i);

	for (int i = 0; i < node->mNumChildren; i++)
		storage[index].childIndices.push_back(traverseNodeTree(storage, node->mChildren[i], counter));

	return index;
}

AssimpObject::AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const mat4 &matrix, int material)
{
	std::string directory;
	const size_t last_slash_idx = filename.rfind('/');
	if (std::string::npos != last_slash_idx)
		directory = filename.substr(0, last_slash_idx);

	Assimp::Importer importer = {};

	const aiScene *scene = importer.ReadFile(filename.data(), (uint)aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
																  aiProcess_CalcTangentSpace | aiProcess_GenUVCoords | aiProcess_FindInstances |
																  aiProcess_RemoveRedundantMaterials);
	if (!scene)
	{
		const std::string error = "Could not load file: " + std::string(filename) + ", error: " + std::string(importer.GetErrorString());
		WARNING(error.c_str());
		throw LoadException(error);
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

	rfw::MeshSkin meshSkin = {};

	std::map<std::string, int> skinNodes = {};
	std::map<std::string, std::vector<int>> animNodeMeshes = {};

	std::map<std::string, int> nodeIndexMapping = {};
	std::vector<NodeWithIndex> sceneNodes(calculateNodeCount(scene->mRootNode));

	int counter = 0;
	traverseNodeTree(sceneNodes.data(), scene->mRootNode, counter);

	for (int i = 0; i < sceneNodes.size(); i++)
		nodeIndexMapping[sceneNodes.at(i).name] = i;

	auto meshes = std::vector<std::vector<rfw::TmpPrim>>(scene->mNumMeshes);
	for (int i = 0; i < scene->mNumMeshes; i++)
	{
		const aiMesh *mesh = scene->mMeshes[i];

		TmpPrim prim = {};

		for (int j = 0; j < mesh->mNumVertices; j++)
		{
			prim.vertices.push_back(glm::make_vec3(&mesh->mVertices[j].x));
			prim.normals.push_back(glm::make_vec3(&mesh->mNormals[j].x));
			if (mesh->HasTextureCoords(0))
				prim.uvs.push_back(glm::make_vec2(&mesh->mTextureCoords[0][j].x));
			else
				prim.uvs.emplace_back(0);
		}

		for (int j = 0; j < mesh->mNumFaces; j++)
		{
			const auto &face = mesh->mFaces[j];
			assert(face.mNumIndices == 3);

			for (int k = 0; k < 3; k++)
				prim.indices.push_back(face.mIndices[k]);
		}

		if (mesh->HasBones())
		{
			prim.joints.resize(prim.vertices.size(), uvec4(0));
			prim.weights.resize(prim.vertices.size(), vec4(0));

			for (int j = 0; j < mesh->mNumBones; j++)
			{
				const auto &bone = mesh->mBones[j];
				const std::string key = bone->mName.C_Str();
				if (animNodeMeshes.find(key) == animNodeMeshes.end())
					animNodeMeshes[key] = {};
				animNodeMeshes[key].push_back(i);
			}

			for (int j = 0; j < mesh->mNumBones; j++)
			{
				const auto &bone = mesh->mBones[j];

				if (skinNodes.find(std::string(bone->mName.C_Str())) == skinNodes.end())
				{
					const std::string name = bone->mName.C_Str();
					const int index = nodeIndexMapping[name];

					meshSkin.jointNodes.push_back(index);
					const int matrixIdx = meshSkin.inverseBindMatrices.size();
					skinNodes[name] = matrixIdx;
					mat4 boneMatrix = mat4(1.0f);
					memcpy(value_ptr(boneMatrix), &bone->mOffsetMatrix, 16 * sizeof(float));
					boneMatrix = rowMajor4(boneMatrix);
					meshSkin.inverseBindMatrices.push_back(boneMatrix);
				}

				for (int k = 0; k < bone->mNumWeights; k++)
				{
					const auto &weight = bone->mWeights[k].mWeight;
					const auto &vID = bone->mWeights[k].mVertexId;

					if (weight <= 0)
						continue;

					auto &joint = prim.joints.at(vID);
					auto &weights = prim.weights.at(vID);

					for (int l = 0; l < 4; l++)
					{
						if (weights[l] == 0)
						{
							joint[l] = skinNodes[std::string(bone->mName.C_Str())];
							weights[l] = weight;
							break;
						}
					}
				}
			}
		}

		prim.poses.resize(mesh->mNumAnimMeshes + 1);
		prim.poses.at(0).positions = prim.vertices;
		prim.poses.at(0).normals = prim.normals;

		for (int j = 0; j < mesh->mNumAnimMeshes; j++)
		{
			auto &pose = prim.poses.at(j + 1);
			const auto animMesh = mesh->mAnimMeshes[j];
			if (!animMesh->HasPositions())
				pose.positions = prim.vertices;
			else
			{
				for (int k = 0; k < animMesh->mNumVertices; k++)
					pose.positions.push_back(glm::make_vec3(&animMesh->mVertices[k].x));
			}

			if (!animMesh->HasNormals())
				pose.normals = prim.normals;
			{
				for (int k = 0; k < animMesh->mNumVertices; k++)
					pose.normals.push_back(glm::make_vec3(&animMesh->mNormals[k].x));
			}

			// TODO: Add texture coordinates, colors, weight
		}

		prim.matID = matMapping[mesh->mMaterialIndex];
		meshes.at(i).push_back(prim);
	}

	for (const auto &node : sceneNodes)
	{
		if (animNodeMeshes.find(node.name) != animNodeMeshes.end())
			object.nodes.push_back(rfw::SceneNode(&object, node.name, node.childIndices, node.meshIDs, {0}, meshes, {}, node.transform));
		else
			object.nodes.push_back(rfw::SceneNode(&object, node.name, node.childIndices, node.meshIDs, {}, meshes, {}, node.transform));
	}

	for (int i = 0; i < scene->mNumAnimations; i++)
	{
		const auto anim = scene->mAnimations[i];

		rfw::SceneAnimation animation = {};
		animation.object = &object;
		animation.duration = anim->mDuration;
		animation.ticksPerSecond = anim->mTicksPerSecond;

		for (int j = 0; j < anim->mNumChannels; j++)
		{
			const auto chan = anim->mChannels[j];

			rfw::SceneAnimation::Channel channel = {};
			channel.nodeIdx = nodeIndexMapping[chan->mNodeName.C_Str()];

			if (chan->mNumPositionKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0; k < chan->mNumPositionKeys; k++)
				{
					const auto posKey = chan->mPositionKeys[k];
					sampler.key_frames.push_back(posKey.mTime);
					sampler.vec3_key.push_back(glm::make_vec3(&posKey.mValue.x));
				}

				const auto samplerID = animation.samplers.size();
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::TRANSLATION);
				animation.samplers.push_back(sampler);
			}

			if (chan->mNumRotationKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0; k < chan->mNumRotationKeys; k++)
				{
					const auto rotKey = chan->mRotationKeys[k];
					sampler.key_frames.push_back(rotKey.mTime);
					sampler.quat_key.emplace_back(rotKey.mValue.w, rotKey.mValue.x, rotKey.mValue.y, rotKey.mValue.z);
				}

				const auto samplerID = animation.samplers.size();
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::ROTATION);
				animation.samplers.push_back(sampler);
			}

			if (chan->mNumScalingKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0; k < chan->mNumScalingKeys; k++)
				{
					const auto scaleKey = chan->mScalingKeys[k];
					sampler.key_frames.push_back(scaleKey.mTime);
					sampler.vec3_key.push_back(glm::make_vec3(&scaleKey.mValue.x));
				}

				const auto samplerID = animation.samplers.size();
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::SCALE);
				animation.samplers.push_back(sampler);
			}

			animation.channels.emplace_back(channel);
		}

		object.animations.emplace_back(animation);
	}

	object.vertices.resize(object.baseVertices.size(), vec4(0, 0, 0, 1));
	object.normals.resize(object.baseNormals.size(), vec3(0.0f));

	object.transformTo(0.0f);
	object.updateTriangles();

	// Update triangle data that only has to be calculated once
	object.updateTriangles(matList);
}

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

	const aiScene *scene = importer.ReadFile(filename.data(), (uint)aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices | aiProcess_Triangulate |
																  aiProcess_CalcTangentSpace | aiProcess_GenUVCoords | aiProcess_FindInstances |
																  aiProcess_RemoveRedundantMaterials);

	if (!scene)
	{
		const std::string error = "Could not load file: " + std::string(filename) + ", error: " + std::string(importer.GetErrorString());
		WARNING(error.c_str());
		throw LoadException(error);
	}

	m_IsAnimated = scene->HasAnimations();

	// Create our own scene graph from Assimp node graph, store every node's transformation along the way
	traverseNode(scene->mRootNode, -1, &m_SceneGraph, &m_NodeNameMapping);
	m_SceneGraph.at(0).update(m_SceneGraph, glm::mat4(1.0f));

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
					channel.positionKeys.at(j) = {
						nodeAnim->mPositionKeys[j].mTime,
						vec3(nodeAnim->mPositionKeys[j].mValue.x, nodeAnim->mPositionKeys[j].mValue.y, nodeAnim->mPositionKeys[j].mValue.z)};

				for (uint j = 0; j < nodeAnim->mNumRotationKeys; j++)
				{
					const aiQuatKey &key = nodeAnim->mRotationKeys[j];
					// const glm::quat q = glm::quat(key.mValue.w, key.mValue.x, key.mValue.y, key.mValue.z);
					channel.rotationKeys.at(j) = {key.mTime, key.mValue};
				}

				for (uint j = 0; j < nodeAnim->mNumScalingKeys; j++)
					channel.scalingKeys.at(j) = {nodeAnim->mScalingKeys[j].mTime, vec3(nodeAnim->mScalingKeys[j].mValue.x, nodeAnim->mScalingKeys[j].mValue.y,
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

	m_TexCoords.resize(vertexCount);

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
				if (curMesh->HasTextureCoords(0))
					m_TexCoords.at(vIdx) = vec2(curMesh->mTextureCoords[0][v].x, curMesh->mTextureCoords[0][v].y);
				else
					m_TexCoords.at(vIdx) = vec2(0.0f);
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
					bone.vertexIDs.resize(aiBone->mNumWeights);
					bone.weights.resize(aiBone->mNumWeights);
					for (uint w = 0; w < aiBone->mNumWeights; w++)
					{
						bone.vertexIDs.at(w) = aiBone->mWeights[w].mVertexId;
						bone.weights.at(w) = aiBone->mWeights[w].mWeight;
					}

					bone.offsetMatrix = glm::make_mat4(&aiBone->mOffsetMatrix[0][0]);
					bone.offsetMatrix = rowMajor4(bone.offsetMatrix);
				}

				m.weights.resize(curMesh->mNumVertices);
				m.joints.resize(curMesh->mNumVertices);
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

	updateTriangles(m_TexCoords);

	// Transform meshes according to node transformations in scene graph
	AssimpObject::transformTo(0.0f);

	// Update triangle data that only has to be calculated once
	for (auto &tri : m_Triangles)
	{
		HostMaterial mat = matList->get(tri.material);

		int texID = mat.map[0].textureID;
		if (texID > -1)
		{
			const Texture &texture = matList->getTextures().at(texID);

			const float Ta =
				static_cast<float>(texture.width * texture.height) * abs((tri.u1 - tri.u0) * (tri.v2 - tri.v0) - (tri.u2 - tri.u0) * (tri.v1 - tri.v0));
			const float Pa = length(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			tri.LOD = 0.5f * log2f(Ta / Pa);
		}
	}

	std::vector<int> skinNodes;
	for (const auto &mesh : m_Meshes)
	{
		if (mesh.bones.empty())
			continue;

		for (const auto &bone : mesh.bones)
			skinNodes.push_back(bone.nodeIndex);
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

	// Start of with clean base data
	m_CurrentVertices.resize(m_BaseVertices.size());
	m_CurrentNormals.resize(m_BaseNormals.size());
	memset(m_CurrentVertices.data(), 0, m_CurrentVertices.size() * sizeof(vec4));
	memset(m_CurrentNormals.data(), 0, m_CurrentNormals.size() * sizeof(vec3));

	if (m_IsAnimated)
	{
		const float timeInTicks = timeInSeconds * static_cast<float>(m_Animations[0].ticksPerSecond);
		const float animationTime = fmod(timeInTicks, static_cast<float>(m_Animations[0].duration));

#if ANIMATION_ENABLED
		// TODO: Implement mesh animations and mesh morph animations
		// Replace node transformations with animation transformations
		for (const auto &animation : m_Animations)
		{
			for (const auto &channel : animation.channels)
			{
				m_SceneGraph.at(channel.nodeIndex).localTransform = channel.getInterpolatedTRS(animationTime);
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

		m_SceneGraph.at(0).update(m_SceneGraph, glm::mat4(1.0f));

#endif
	}

	for (const auto &mesh : m_Meshes)
	{
		// Set current data for this mesh to base data
		if (mesh.bones.empty())
		{
			/*
			 * Unfortunately, this part of the loop cannot be done on initialization.
			 * Meshes that are affected by animations can be placed in a node with a parent hierarchy
			 * in which one of the parent nodes can be influenced by an animations.
			 */
			const aligned_mat4 transform = rowMajor4(m_SceneGraph.at(mesh.nodeIndex).combinedTransform);
			const aligned_mat3 transform3x3 = glm::mat3(transform);

			for (uint i = 0, vIdx = mesh.vertexOffset; i < mesh.vertexCount; i++, vIdx++)
			{
				const glm::aligned_vec4 baseVertex = m_BaseVertices.at(vIdx);
				const glm::aligned_vec3 baseNormal = m_BaseNormals.at(vIdx);

				m_CurrentVertices.at(vIdx) = transform * baseVertex;
				m_CurrentNormals.at(vIdx) = transform3x3 * baseNormal;
			}
			continue;
		}

#if 1
		for (const auto &bone : mesh.bones)
		{
			const glm::mat4 &skin4x4 = m_SceneGraph.at(bone.nodeIndex).combinedTransform * bone.offsetMatrix;
			const glm::mat3 skin3x3 = mat3(skin4x4);

			for (int i = 0, s = int(bone.vertexIDs.size()); i < s; i++)
			{
				const uint vIdx = mesh.vertexOffset + bone.vertexIDs.at(i);
				m_CurrentVertices.at(vIdx) += skin4x4 * m_BaseVertices.at(vIdx) * bone.weights.at(i);
				m_CurrentNormals.at(vIdx) += skin3x3 * m_BaseNormals.at(vIdx) * bone.weights.at(i);
			}
		}
#else
		std::vector<mat4> skinMatrices;

		for (int i = 0, s = mesh.vertexCount; i < s; i++)
		{
			const uvec4 &j4 = mesh.joints.at(i);
			const vec4 &w4 = mesh.weights.at(i);

			const mat4 mx = skinMatrices.at(j4.x) * w4.x;
			const mat4 my = skinMatrices.at(j4.y) * w4.y;
			const mat4 mz = skinMatrices.at(j4.z) * w4.z;
			const mat4 mw = skinMatrices.at(j4.w) * w4.w;

			const mat4 skinMatrix = mx + my + mz + mw;

			m_CurrentVertices[i + mesh.vertexOffset] = skinMatrix * m_BaseVertices[i + mesh.vertexOffset];
			m_CurrentNormals[i + mesh.vertexOffset] = skinMatrix * vec4(m_BaseNormals[i + mesh.vertexOffset], 0);
		}
#endif
	}

	updateTriangles();
}

void AssimpObject::updateTriangles()
{
	m_Triangles.resize(m_Indices.size());
	for (size_t i = 0, s = m_Triangles.size(); i < s; i++)
	{
		Triangle &tri = m_Triangles.at(i);
		const glm::uvec3 &indices = m_Indices.at(i);
		const vec3 &n0 = m_CurrentNormals.at(indices.x);
		const vec3 &n1 = m_CurrentNormals.at(indices.y);
		const vec3 &n2 = m_CurrentNormals.at(indices.z);

		tri.vertex0 = m_CurrentVertices.at(indices.x);
		tri.vertex1 = m_CurrentVertices.at(indices.y);
		tri.vertex2 = m_CurrentVertices.at(indices.z);

		vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
		if (dot(N, n0) < 0.0f && dot(N, n1) < 0.0f && dot(N, n2) < 0.0f)
			N *= -1.0f; // flip if not consistent with vertex normals

		tri.Nx = N.x;
		tri.Ny = N.y;
		tri.Nz = N.z;

		tri.vN0 = n0;
		tri.vN1 = n1;
		tri.vN2 = n2;
	}
}

void rfw::AssimpObject::updateTriangles(const std::vector<glm::vec2> &uvs)
{
	m_Triangles.resize(m_Indices.size());
	for (size_t i = 0, s = m_Triangles.size(); i < s; i++)
	{
		Triangle &tri = m_Triangles.at(i);
		const glm::uvec3 &indices = m_Indices.at(i);

		const vec3 &n0 = m_CurrentNormals.at(indices.x);
		const vec3 &n1 = m_CurrentNormals.at(indices.y);
		const vec3 &n2 = m_CurrentNormals.at(indices.z);

		tri.vertex0 = m_CurrentVertices.at(indices.x);
		tri.vertex1 = m_CurrentVertices.at(indices.y);
		tri.vertex2 = m_CurrentVertices.at(indices.z);

		const vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));

		tri.u0 = uvs.at(indices.x).x;
		tri.v0 = uvs.at(indices.x).y;

		tri.u1 = uvs.at(indices.y).x;
		tri.v1 = uvs.at(indices.y).y;

		tri.u2 = uvs.at(indices.z).x;
		tri.v2 = uvs.at(indices.z).y;

		tri.Nx = N.x;
		tri.Ny = N.y;
		tri.Nz = N.z;

		tri.vN0 = n0;
		tri.vN1 = n1;
		tri.vN2 = n2;

		tri.material = m_MaterialIndices.at(i);
	}
}

size_t rfw::AssimpObject::traverseNode(const aiNode *node, int parentIdx, std::vector<AssimpNode> *storage, std::map<std::string, uint> *nodeNameMapping)
{
	// Get current index
	const size_t currentNodeIndex = storage->size();
	// Add index to current node to name mapping
	(*nodeNameMapping)[std::string(node->mName.C_Str())] = static_cast<uint>(currentNodeIndex);

	// Initialize node data
	AssimpNode n;
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

	memcpy(value_ptr(n.localTransform), &node->mTransformation[0][0], sizeof(mat4));
	n.localTransform = rowMajor4(n.localTransform);
	storage->push_back(n);

	// Iterate and initialize children
	const int currentIdx = static_cast<int>(currentNodeIndex);
	for (uint i = 0; i < node->mNumChildren; i++)
	{
		const aiNode *child = node->mChildren[i];
		const size_t childIdx = traverseNode(child, currentIdx, storage, nodeNameMapping);
		storage->at(currentNodeIndex).children.at(i) = static_cast<uint>(childIdx);
	}

	return currentNodeIndex;
}

uint AssimpObject::getAnimationCount() const { return static_cast<uint>(m_Animations.size()); }

void AssimpObject::setAnimation(uint index) { m_CurrentAnimation = clamp(index, 0u, static_cast<uint>(m_Animations.size())); }

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

rfw::Mesh rfw::AssimpObject::getMesh() const
{
#if 0
	rfw::Mesh mesh{};
	mesh.vertices = m_CurrentVertices.data();
	mesh.normals = m_CurrentNormals.data();
	mesh.triangles = m_Triangles.data();
	mesh.vertexCount = m_CurrentVertices.size();
	mesh.triangleCount = m_Indices.size();
	mesh.indices = m_Indices.data();
	mesh.texCoords = m_TexCoords.data();
	return mesh;
#else
	auto mesh = rfw::Mesh();
	mesh.vertices = object.vertices.data();
	mesh.normals = object.normals.data();
	mesh.triangles = object.triangles.data();
	if (object.indices.empty())
	{
		mesh.indices = nullptr;
		mesh.triangleCount = object.vertices.size() / 3;
	}
	else
	{
		mesh.indices = object.indices.data();
		mesh.triangleCount = object.indices.size();
	}
	mesh.vertexCount = object.vertices.size();
	mesh.texCoords = object.texCoords.data();
	return mesh;
#endif
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
	auto result = glm::identity<glm::mat4>();
	int keyIndex = 0;
	float deltaTime, factor;

	for (int i = 0, s = int(positionKeys.size()) - 1; i < s; i++)
	{
		if (time < positionKeys.at(i).time)
		{
			keyIndex = i;
			break;
		}
	}

	keyIndex = keyIndex % (positionKeys.size() - 1);

	assert(keyIndex < positionKeys.size());
	deltaTime = positionKeys.at(keyIndex + 1).time - positionKeys.at(keyIndex).time;
	factor = (time - positionKeys.at(keyIndex).time) / deltaTime;
	const vec3 &startPos = positionKeys.at(keyIndex).value;
	const vec3 &endPos = positionKeys.at(keyIndex + 1).value;
	result = glm::translate(result, startPos + factor * (endPos - startPos));

	for (int i = 0, s = int(rotationKeys.size()) - 1; i < s; i++)
	{
		if (time < rotationKeys.at(i).time)
		{
			keyIndex = i;
			break;
		}
	}
	assert(keyIndex < rotationKeys.size());
	deltaTime = rotationKeys.at(keyIndex + 1).time - rotationKeys.at(keyIndex).time;
	factor = (time - rotationKeys.at(keyIndex).time) / deltaTime;
	const auto &startQuat = rotationKeys.at(keyIndex).value;
	const auto &endQuat = rotationKeys.at(keyIndex + 1).value;
	const glm::quat q = glm::mix(startQuat, endQuat, factor);
	result = result * glm::mat4_cast(glm::quat(q.w, q.x, q.y, q.z));

	for (int i = 0, s = int(scalingKeys.size()) - 1; i < s; i++)
	{
		if (time < scalingKeys.at(i).time)
		{
			keyIndex = i;
			break;
		}
	}

	assert(keyIndex < scalingKeys.size());
	deltaTime = scalingKeys.at(keyIndex + 1).time - scalingKeys.at(keyIndex).time;
	factor = (time - scalingKeys.at(keyIndex).time) / deltaTime;
	const vec3 &startScale = scalingKeys.at(keyIndex).value;
	const vec3 &endScale = scalingKeys.at(keyIndex + 1).value;
	result = result * glm::scale(glm::mat4(1.0f), startScale + factor * (endScale - startScale));

	return result;
}

void AssimpObject::AssimpNode::update(std::vector<AssimpNode> &nodes, const glm::mat4 &T)
{
	combinedTransform = T * localTransform;

	for (const auto child : children)
		nodes.at(child).update(nodes, combinedTransform);
}