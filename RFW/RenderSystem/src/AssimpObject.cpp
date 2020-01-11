#include "rfw.h"
#include "Internal.h"

#define USE_PARALLEL_FOR 1

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
	for (int i = 0, s = static_cast<int>(node->mNumChildren); i < s; i++)
		count += calculateNodeCount(node->mChildren[i]);
	return count;
}

int traverseNodeTree(NodeWithIndex *storage, const aiNode *node, int &counter)
{
	const int index = counter++;

	storage[index].name = node->mName.C_Str();
	storage[index].node = node;
	storage[index].index = index;

	for (int i = 0, s = static_cast<int>(node->mNumMeshes); i < s; i++)
		storage[index].meshIDs.push_back(node->mMeshes[i]);

	mat4 T;
	memcpy(value_ptr(T), &node->mTransformation, 16 * sizeof(float));
	T = rowMajor4(T);
	storage[index].transform = T;

	for (int i = 0, s = static_cast<int>(node->mNumMeshes); i < s; i++)
		storage[index].meshIDs.push_back(i);

	for (int i = 0, s = static_cast<int>(node->mNumChildren); i < s; i++)
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
	for (int i = 0, s = static_cast<int>(scene->mNumMeshes); i < s; i++)
	{
		const aiMesh *mesh = scene->mMeshes[i];

		TmpPrim prim = {};

		for (int j = 0, sj = static_cast<int>(mesh->mNumVertices); j < sj; j++)
		{
			prim.vertices.push_back(glm::make_vec3(&mesh->mVertices[j].x));
			prim.normals.push_back(glm::make_vec3(&mesh->mNormals[j].x));
			if (mesh->HasTextureCoords(0))
				prim.uvs.push_back(glm::make_vec2(&mesh->mTextureCoords[0][j].x));
			else
				prim.uvs.emplace_back(0.0f, 0.0f);
		}
		for (int j = 0, sj = static_cast<int>(mesh->mNumFaces); j < sj; j++)
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

			for (int j = 0, sj = static_cast<int>(mesh->mNumBones); j < sj; j++)
			{
				const auto &bone = mesh->mBones[j];
				const std::string key = bone->mName.C_Str();
				if (animNodeMeshes.find(key) == animNodeMeshes.end())
					animNodeMeshes[key] = {};
				animNodeMeshes[key].push_back(i);
			}

			for (int j = 0, sj = static_cast<int>(mesh->mNumBones); j < sj; j++)
			{
				const auto &bone = mesh->mBones[j];

				if (skinNodes.find(std::string(bone->mName.C_Str())) == skinNodes.end())
				{
					const std::string name = bone->mName.C_Str();
					const int index = nodeIndexMapping[name];

					meshSkin.jointNodes.push_back(index);
					const int matrixIdx = static_cast<int>(meshSkin.inverseBindMatrices.size());
					skinNodes[name] = matrixIdx;
					mat4 boneMatrix = mat4(1.0f);
					memcpy(value_ptr(boneMatrix), &bone->mOffsetMatrix, 16 * sizeof(float));
					boneMatrix = rowMajor4(boneMatrix);
					meshSkin.inverseBindMatrices.push_back(boneMatrix);
				}

				for (int k = 0, sk = static_cast<int>(bone->mNumWeights); k < sk; k++)
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

		for (int j = 0, sj = static_cast<int>(mesh->mNumAnimMeshes); j < sj; j++)
		{
			auto &pose = prim.poses.at(j + 1);
			const auto animMesh = mesh->mAnimMeshes[j];
			if (!animMesh->HasPositions())
				pose.positions = prim.vertices;
			else
			{
				for (int k = 0, sk = static_cast<int>(animMesh->mNumVertices); k < sk; k++)
					pose.positions.push_back(glm::make_vec3(&animMesh->mVertices[k].x));
			}

			if (!animMesh->HasNormals())
				pose.normals = prim.normals;
			{
				for (int k = 0, sk = static_cast<int>(animMesh->mNumVertices); k < sk; k++)
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

	if (scene->HasAnimations())
	{
		meshSkin.jointMatrices.resize(meshSkin.inverseBindMatrices.size(), glm::mat4(1.0f));
		object.skins.push_back(meshSkin);
	}

	for (int i = 0, si = static_cast<int>(scene->mNumAnimations); i < si; i++)
	{
		const auto anim = scene->mAnimations[i];

		rfw::SceneAnimation animation = {};
		animation.object = &object;
		animation.duration = anim->mDuration;
		animation.ticksPerSecond = anim->mTicksPerSecond;

		for (int j = 0, sj = static_cast<int>(anim->mNumChannels); j < sj; j++)
		{
			const auto chan = anim->mChannels[j];

			rfw::SceneAnimation::Channel channel = {};
			channel.nodeIdx = nodeIndexMapping[chan->mNodeName.C_Str()];

			if (chan->mNumPositionKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0, sk = static_cast<int>(chan->mNumPositionKeys); k < sk; k++)
				{
					const auto posKey = chan->mPositionKeys[k];
					sampler.key_frames.push_back(static_cast<float>(posKey.mTime));
					sampler.vec3_key.push_back(glm::make_vec3(&posKey.mValue.x));
				}

				const auto samplerID = static_cast<int>(animation.samplers.size());
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::TRANSLATION);
				animation.samplers.push_back(sampler);
			}

			if (chan->mNumRotationKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0, sk = static_cast<int>(chan->mNumRotationKeys); k < sk; k++)
				{
					const auto rotKey = chan->mRotationKeys[k];
					sampler.key_frames.push_back(static_cast<float>(rotKey.mTime));
					sampler.quat_key.emplace_back(rotKey.mValue.w, rotKey.mValue.x, rotKey.mValue.y, rotKey.mValue.z);
				}

				const auto samplerID = static_cast<int>(animation.samplers.size());
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::ROTATION);
				animation.samplers.push_back(sampler);
			}

			if (chan->mNumScalingKeys > 0)
			{
				rfw::SceneAnimation::Sampler sampler = {};
				sampler.method = rfw::SceneAnimation::Sampler::LINEAR;
				for (int k = 0, sk = static_cast<int>(chan->mNumScalingKeys); k < sk; k++)
				{
					const auto scaleKey = chan->mScalingKeys[k];
					sampler.key_frames.push_back(static_cast<float>(scaleKey.mTime));
					sampler.vec3_key.push_back(glm::make_vec3(&scaleKey.mValue.x));
				}

				const auto samplerID = static_cast<int>(animation.samplers.size());
				channel.samplerIDs.push_back(samplerID);
				channel.targets.push_back(rfw::SceneAnimation::Channel::SCALE);
				animation.samplers.push_back(sampler);
			}

			animation.channels.emplace_back(channel);
		}

		object.animations.emplace_back(animation);
	}

	for (auto &node : object.nodes)
	{
		if (animNodeMeshes.find(node.name) != animNodeMeshes.end())
		{
			for (int i = 0; i < node.meshIDs.size(); i++)
				node.skinIDs.at(i) = 0;
		}
	}

	object.vertices.resize(object.baseVertices.size(), vec4(0, 0, 0, 1));
	object.normals.resize(object.baseNormals.size(), vec3(0.0f));

	object.nodes.at(0).update(mat4(1.0f));

	object.transformTo(0.0f);

	object.updateTriangles();

	// Update triangle data that only has to be calculated once
	object.updateTriangles(matList);

	utils::logger::log("Loaded file: %s with %u vertices and %u triangles", filename.data(), object.vertices.size(), object.triangles.size());
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

	if (!scene || scene->mNumMeshes == 0)
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

	if (m_NodesWithMeshes.empty())
	{
		m_RfwMeshes.resize(scene->mNumMeshes);
		m_Meshes.resize(scene->mNumMeshes);
		m_SceneGraph.resize(scene->mNumMeshes);
		for (int i = 0, si = static_cast<int>(scene->mNumMeshes); i < si; i++)
		{
			m_SceneGraph[i].localTransform = glm::mat4(1.0f);
			m_SceneGraph[i].combinedTransform = glm::mat4(1.0f);
			m_SceneGraph[i].meshes.push_back(i);
		}

		for (int i = 0, si = static_cast<int>(scene->mNumMeshes); i < si; i++)
		{
			auto *aiMesh = scene->mMeshes[i];

			m_BaseVertices.reserve(m_BaseVertices.size() + aiMesh->mNumVertices);
			m_BaseNormals.reserve(m_BaseVertices.size() + aiMesh->mNumVertices);
			m_TexCoords.reserve(m_BaseVertices.size() + aiMesh->mNumVertices);

			m_Indices.resize(m_Indices.size() + aiMesh->mNumFaces);
			m_MaterialIndices.resize(m_MaterialIndices.size() + aiMesh->mNumFaces);

			auto &mesh = m_Meshes[i];
			mesh.nodeIndex = i;
			mesh.materialIdx = matMapping[aiMesh->mMaterialIndex];
			mesh.vertexCount = aiMesh->mNumVertices;
			mesh.vertexOffset = static_cast<uint>(m_BaseVertices.size());
			mesh.faceCount = aiMesh->mNumFaces;
			mesh.faceOffset = static_cast<uint>(m_Indices.size());

			for (int i = 0, s = static_cast<int>(aiMesh->mNumVertices); i < s; i++)
			{
				m_BaseVertices.emplace_back(aiMesh->mVertices[i].x, aiMesh->mVertices[i].y, aiMesh->mVertices[i].z, 1.0f);
				m_BaseNormals.emplace_back(aiMesh->mNormals[i].x, aiMesh->mNormals[i].y, aiMesh->mNormals[i].z, 0.0f);

				if (aiMesh->HasTextureCoords(0))
					m_TexCoords.emplace_back(aiMesh->mTextureCoords[0][i].x, aiMesh->mTextureCoords[0][i].y);
				else
					m_TexCoords.emplace_back(0.0f);
			}

			for (int i = 0, s = static_cast<int>(aiMesh->mNumFaces); i < s; i++)
			{
				m_MaterialIndices.emplace_back(matMapping[aiMesh->mMaterialIndex]);
				m_Indices.emplace_back(aiMesh->mFaces[i].mIndices[0], aiMesh->mFaces[i].mIndices[1], aiMesh->mFaces[i].mIndices[2]);
			}
		}
	}
	else
	{
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
			for (unsigned int meshIdx : node.meshes)
			{
				m_MeshMapping.at(meshIdx).push_back(meshIndex);

				MeshInfo &m = m_Meshes.at(meshIndex);
				const aiMesh *curMesh = scene->mMeshes[meshIdx];
				const uint matIndex = matMapping[curMesh->mMaterialIndex];

				for (uint v = 0; v < curMesh->mNumVertices; v++)
				{
					const auto vIdx = vertexOffset + v;

					const vec4 vertex = vec4(glm::make_vec3(&curMesh->mVertices[v].x), 1.0f);
					const vec3 normal = glm::make_vec3(&curMesh->mNormals[v].x);

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
					m_Indices.at(fIdx) = uvec3(f0, f1, f2);
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

						const aiNode *boneNode = scene->mRootNode->FindNode(aiBone->mName);
						bone.nodeIndex = m_NodeNameMapping[std::string(boneNode->mName.C_Str())];
						bone.nodeName = std::string(boneNode->mName.C_Str());

						// Store bone weights
						bone.vertexIDs.resize(aiBone->mNumWeights);
						bone.weights.resize(aiBone->mNumWeights);
						for (uint w = 0; w < aiBone->mNumWeights; w++)
						{
							bone.vertexIDs.at(w) = aiBone->mWeights[w].mVertexId;
							bone.weights.at(w) = aiBone->mWeights[w].mWeight;
						}

						bone.offsetMatrix = glm::make_mat4(&aiBone->mOffsetMatrix[0][0]);
						bone.offsetMatrix.matrix = rowMajor4(bone.offsetMatrix.matrix);
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
	}

	const simd::vector4 normal_mask = _mm_set_epi32(0, ~0, ~0, ~0);
	m_CurrentVertices.resize(m_BaseVertices.size());
	m_CurrentNormals.resize(m_BaseNormals.size());
	memcpy(m_CurrentVertices.data(), m_BaseVertices.data(), m_BaseVertices.size() * sizeof(vec4));
	for (int i = 0, s = static_cast<int>(m_BaseNormals.size()); i < s; i++)
		m_BaseNormals[i].write_to(value_ptr(m_CurrentNormals[i]), normal_mask);

	if (hasTransform)
		m_SceneGraph[0].localTransform = matrix * m_SceneGraph[0].localTransform.matrix;

	// Transform meshes according to node transformations in scene graph
	AssimpObject::transformTo(0.0f);

	updateTriangles(m_TexCoords);

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
			tri.LOD = max(0.f, sqrt(0.5f * log2f(Ta / Pa)));
		}
	}

	m_MeshTransforms.resize(m_Meshes.size(), glm::mat4(1.0f));

	std::vector<int> skinNodes;
	for (int meshID = 0, s = static_cast<int>(m_Meshes.size()); meshID < s; meshID++)
	{
		const auto &mesh = m_Meshes[meshID];

		if (mesh.bones.empty())
			continue;

		for (const auto &bone : mesh.bones)
			skinNodes.push_back(bone.nodeIndex);
	}

	DEBUG("Loaded file: %s with %u vertices and %u triangles", filename.data(), m_CurrentVertices.size(), m_Triangles.size());
}

void AssimpObject::transformTo(const float timeInSeconds)
{
	const simd::vector4 normal_mask = _mm_set_epi32(0, ~0, ~0, ~0);

	if (!object.vertices.empty())
	{
		object.transformTo(timeInSeconds);
		return;
	}

	if (!m_IsAnimated && m_HasUpdated)
		return;

	if (!m_IsAnimated)
	{
		m_CurrentVertices.resize(m_BaseVertices.size());
		m_CurrentNormals.resize(m_BaseNormals.size());
		memcpy(m_CurrentVertices.data(), m_BaseVertices.data(), m_BaseVertices.size() * sizeof(vec4));
		for (int i = 0, s = static_cast<int>(m_BaseNormals.size()); i < s; i++)
		{
			// m_BaseNormals[i].write_to(value_ptr(m_CurrentNormals[i]), normal_mask);
			m_CurrentNormals[i] = m_BaseNormals[i].vec;
		}

		m_BaseVertices.clear();
		m_BaseNormals.clear();

		// Start of with clean base data
		m_MeshTransforms.resize(m_Meshes.size(), glm::mat4(1.0f));
		m_HasUpdated = true;

		m_SceneGraph[0].update(m_SceneGraph, glm::mat4(1.0f));
		for (int meshID = 0, s = static_cast<int>(m_Meshes.size()); meshID < s; meshID++)
			m_MeshTransforms[meshID] = m_SceneGraph[m_Meshes[meshID].nodeIndex].combinedTransform;
		return;
	}

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
				m_SceneGraph[channel.nodeIndex].localTransform = channel.getInterpolatedTRS(animationTime);

			for (const auto &channel : animation.meshChannels)
			{
				// TODO
			}

			for (const auto &channel : animation.morphChannels)
			{
				// TODO
			}
		}

		m_SceneGraph[0].update(m_SceneGraph, glm::mat4(1.0f));
#endif
	}

	m_ChangedMeshTransforms.resize(m_Meshes.size(), false);
	m_MeshTransforms.resize(m_Meshes.size());

#if USE_PARALLEL_FOR
	for (int i = 0, s = static_cast<int>(m_Meshes.size()); i < s; i++)
	{
		m_ChangedMeshTransforms[i] = true;
		auto &mesh = m_Meshes[i];

		m_MeshTransforms[i] = m_SceneGraph[mesh.nodeIndex].combinedTransform.matrix;
		if (mesh.bones.empty())
			continue;

		// m_MeshTransforms[i] = glm::mat4(1.0f);
		memset(m_CurrentVertices.data() + mesh.vertexOffset, 0, mesh.vertexCount * sizeof(vec4));
		memset(m_CurrentNormals.data() + mesh.vertexOffset, 0, mesh.vertexCount * sizeof(vec3));

		// Set current data for this mesh to base data
		for (const auto &bone : mesh.bones)
		{
			const simd::matrix4 skin =
				m_SceneGraph[mesh.nodeIndex].combinedTransform.inversed() * m_SceneGraph[bone.nodeIndex].combinedTransform * bone.offsetMatrix;
			const simd::matrix4 normal_matrix = skin.inversed().transposed();

			for (int i = 0, s = int(bone.vertexIDs.size()); i < s; i++)
			{
				const uint vIdx = mesh.vertexOffset + bone.vertexIDs[i];

				const simd::vector4 &vertex = m_BaseVertices[vIdx];
				const simd::vector4 curVertex = m_CurrentVertices[vIdx];
				const simd::vector4 weight4 = bone.weights[i];

				simd::vector4 result = curVertex + skin * vertex * weight4;
				result.write_to(value_ptr(m_CurrentVertices[vIdx]));

				const auto cur_normal = simd::vector4(value_ptr(m_CurrentNormals[vIdx]), normal_mask);
				const simd::vector4 &normal = m_BaseNormals[vIdx];
				result = cur_normal + normal_matrix * normal * weight4;
				result.write_to(value_ptr(m_CurrentNormals[vIdx]), normal_mask);
			}
		}

		mesh.dirty = true;
	}
#else
	for (int i = 0, s = static_cast<int>(m_Meshes.size()); i < s; i++)
	{
		m_ChangedMeshTransforms[i] = true;
		auto &mesh = m_Meshes[i];

		m_MeshTransforms[i] = m_SceneGraph[mesh.nodeIndex].combinedTransform.matrix;
		if (mesh.bones.empty())
			continue;

		// m_MeshTransforms[i] = glm::mat4(1.0f);
		memset(m_CurrentVertices.data() + mesh.vertexOffset, 0, mesh.vertexCount * sizeof(vec4));
		memset(m_CurrentNormals.data() + mesh.vertexOffset, 0, mesh.vertexCount * sizeof(vec3));

		// Set current data for this mesh to base data
		for (const auto &bone : mesh.bones)
		{
			const simd::matrix4 skin =
				m_SceneGraph[mesh.nodeIndex].combinedTransform.inversed() * m_SceneGraph[bone.nodeIndex].combinedTransform * bone.offsetMatrix;
			const simd::matrix4 normal_matrix = skin.inversed().transposed();

			for (int i = 0, s = int(bone.vertexIDs.size()); i < s; i++)
			{
				const uint vIdx = mesh.vertexOffset + bone.vertexIDs[i];

				const simd::vector4 &vertex = m_BaseVertices[vIdx];
				const simd::vector4 curVertex = m_CurrentVertices[vIdx];
				const simd::vector4 weight4 = bone.weights[i];

				simd::vector4 result = curVertex + skin * vertex * weight4;
				result.write_to(value_ptr(m_CurrentVertices[vIdx]));

				const auto cur_normal = simd::vector4(value_ptr(m_CurrentNormals[vIdx]), normal_mask);
				const simd::vector4 &normal = m_BaseNormals[vIdx];
				result = cur_normal + normal_matrix * normal * weight4;
				result.write_to(value_ptr(m_CurrentNormals[vIdx]), normal_mask);
			}
		}

		mesh.dirty = true;
	}
#endif

	updateTriangles();
}

void AssimpObject::updateTriangles()
{
	m_Triangles.resize(m_Indices.size());

#if USE_PARALLEL_FOR
	rfw::utils::concurrency::parallel_for<int>(0, static_cast<int>(m_Meshes.size()), [&](int i) {
		const auto &mesh = m_Meshes[i];

		for (int i = 0, s = static_cast<int>(mesh.faceCount); i < s; i++)
		{
			const auto triIdx = i + mesh.faceOffset;

			Triangle &tri = m_Triangles.at(triIdx);
			const glm::uvec3 &indices = m_Indices.at(triIdx) + mesh.vertexOffset;

			tri.vN0 = m_CurrentNormals[indices.x];
			tri.vN1 = m_CurrentNormals[indices.y];
			tri.vN2 = m_CurrentNormals[indices.z];

			tri.vertex0 = m_CurrentVertices[indices.x];
			tri.vertex1 = m_CurrentVertices[indices.y];
			tri.vertex2 = m_CurrentVertices[indices.z];

			vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			if (dot(N, tri.vN0) < 0.0f && dot(N, tri.vN1) < 0.0f && dot(N, tri.vN2) < 0.0f)
				N *= -1.0f; // flip if not consistent with vertex normals

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;
		}
	});
#else
	for (const auto &mesh : m_Meshes)
	{
		for (int i = 0, s = static_cast<int>(mesh.faceCount); i < s; i++)
		{
			const auto triIdx = i + mesh.faceOffset;

			Triangle &tri = m_Triangles.at(triIdx);
			const glm::uvec3 &indices = m_Indices.at(triIdx) + mesh.vertexOffset;

			tri.vN0 = m_CurrentNormals[indices.x];
			tri.vN1 = m_CurrentNormals[indices.y];
			tri.vN2 = m_CurrentNormals[indices.z];

			tri.vertex0 = m_CurrentVertices[indices.x];
			tri.vertex1 = m_CurrentVertices[indices.y];
			tri.vertex2 = m_CurrentVertices[indices.z];

			vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			if (dot(N, tri.vN0) < 0.0f && dot(N, tri.vN1) < 0.0f && dot(N, tri.vN2) < 0.0f)
				N *= -1.0f; // flip if not consistent with vertex normals

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;
		}
	}
#endif
}

void rfw::AssimpObject::updateTriangles(const std::vector<glm::vec2> &uvs)
{
	m_Triangles.resize(m_Indices.size());

	rfw::utils::concurrency::parallel_for<int>(0, static_cast<int>(m_Meshes.size()), [&](int i) {
		const auto &mesh = m_Meshes[i];
		for (size_t i = 0, s = mesh.faceCount; i < s; i++)
		{
			const glm::uvec3 &indices = m_Indices.at(i + mesh.faceOffset) + mesh.vertexOffset;
			Triangle &tri = m_Triangles[i + mesh.faceOffset];

			tri.vN0 = m_CurrentNormals[indices.x];
			tri.vN1 = m_CurrentNormals[indices.y];
			tri.vN2 = m_CurrentNormals[indices.z];

			tri.vertex0 = m_CurrentVertices[indices.x];
			tri.vertex1 = m_CurrentVertices[indices.y];
			tri.vertex2 = m_CurrentVertices[indices.z];

			vec3 N = normalize(cross(tri.vertex1 - tri.vertex0, tri.vertex2 - tri.vertex0));
			if (dot(N, tri.vN0) < 0.0f && dot(N, tri.vN1) < 0.0f && dot(N, tri.vN2) < 0.0f)
				N *= -1.0f; // flip if not consistent with vertex normals

			tri.u0 = uvs[indices.x].x;
			tri.v0 = uvs[indices.x].y;

			tri.u1 = uvs[indices.y].x;
			tri.v1 = uvs[indices.y].y;

			tri.u2 = uvs[indices.z].x;
			tri.v2 = uvs[indices.z].y;

			tri.Nx = N.x;
			tri.Ny = N.y;
			tri.Nz = N.z;

			tri.material = m_MaterialIndices[i + mesh.faceOffset];
		}
	});
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
			n.meshes[i] = meshIdx;
		}

		m_NodesWithMeshes.push_back(currentNodeIndex);
	}

	memcpy(value_ptr(n.localTransform.matrix), &node->mTransformation[0][0], sizeof(mat4));
	n.localTransform = rowMajor4(n.localTransform.matrix);
	storage->push_back(n);

	// Iterate and initialize children
	const int currentIdx = static_cast<int>(currentNodeIndex);
	for (uint i = 0; i < node->mNumChildren; i++)
	{
		const aiNode *child = node->mChildren[i];
		const size_t childIdx = traverseNode(child, currentIdx, storage, nodeNameMapping);
		(*storage)[currentNodeIndex].children[i] = static_cast<uint>(childIdx);
	}

	return currentNodeIndex;
}

const std::vector<std::vector<int>> &rfw::AssimpObject::getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize)
{
	// TODO: rewrite to initialize per mesh
	if (reinitialize)
	{
		m_LightIndices.clear();
		m_LightIndices.resize(m_Meshes.size());

		for (int i = 0, s = static_cast<int>(m_Meshes.size()); i < s; i++)
		{
			const auto &mesh = m_Meshes[i];
			std::vector<int> &indices = m_LightIndices[i];

			const size_t offset = indices.size();
			if (matLightFlags.at(mesh.materialIdx))
			{
				indices.reserve(offset + mesh.faceCount);
				for (int i = 0, s = static_cast<int>(mesh.faceCount); i < s; i++)
					indices.push_back(static_cast<int>(mesh.faceOffset) + i);
			}
		}
	}

	return m_LightIndices;
}

const std::vector<std::pair<size_t, rfw::Mesh>> &rfw::AssimpObject::getMeshes() const { return m_RfwMeshes; }

const std::vector<rfw::simd::matrix4> &rfw::AssimpObject::getMeshTransforms() const { return m_MeshTransforms; }

std::vector<bool> rfw::AssimpObject::getChangedMeshes()
{
	assert(m_Meshes.size() == m_RfwMeshes.size());

	auto changed = std::vector<bool>(m_Meshes.size(), false);
	for (int i = 0, s = static_cast<int>(m_Meshes.size()); i < s; i++)
	{
		if (m_Meshes[i].dirty)
		{
			changed[i] = true;
			m_Meshes[i].dirty = false;
		}
	}

	return changed;
}

std::vector<bool> rfw::AssimpObject::getChangedMeshMatrices()
{
	auto values = std::move(m_ChangedMeshTransforms);
	m_ChangedMeshTransforms.resize(m_Meshes.size(), false);
	return values;
}

void rfw::AssimpObject::prepareMeshes(RenderSystem &rs)
{
	m_RfwMeshes.reserve(m_Meshes.size());
	for (const auto &mesh : m_Meshes)
	{
		auto rfwMesh = rfw::Mesh();

		rfwMesh.vertices = &m_CurrentVertices[mesh.vertexOffset];
		rfwMesh.normals = &m_CurrentNormals[mesh.vertexOffset];
		rfwMesh.triangles = &m_Triangles[mesh.faceOffset];
		rfwMesh.indices = &m_Indices[mesh.faceOffset];
		rfwMesh.texCoords = &m_TexCoords[mesh.vertexOffset];

		rfwMesh.vertexCount = mesh.vertexCount;
		rfwMesh.triangleCount = mesh.faceCount;
		m_RfwMeshes.emplace_back(rs.requestMeshIndex(), rfwMesh);
	}
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
	combinedTransform = T * localTransform.matrix;

	for (const auto child : children)
		nodes.at(child).update(nodes, combinedTransform.matrix);
}