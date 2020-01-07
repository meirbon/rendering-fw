#pragma once

#include <MathIncludes.h>

#include <assimp/Importer.hpp>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

#include <exception>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <tuple>
#include <map>

#include <Structures.h>
#include "MaterialList.h"
#include "SceneTriangles.h"

#include "gLTF/SceneObject.h"
#include "gLTF/Skinning.h"

namespace rfw
{
class AssimpObject : public SceneTriangles
{
  private:
	struct AnimationChannel
	{
		struct PositionKey
		{
			PositionKey() = default;
			PositionKey(double t, glm::vec3 v) : time(float(t)), value(v) {}
			float time;
			glm::vec3 value;
		};

		struct RotationKey
		{
			RotationKey() = default;
			RotationKey(double t, aiQuaternion v) : time(float(t)), value(glm::quat(v.w, v.x, v.y, v.z)) {}
			float time;
			glm::quat value;
		};

		struct ScalingKey
		{
			ScalingKey() = default;
			ScalingKey(double t, glm::vec3 v) : time(float(t)), value(v) {}
			float time;
			glm::vec3 value;
		};

		[[nodiscard]] glm::mat4 getInterpolatedTRS(float time) const;

		/** The name of the node affected by this animation. The node
		 *  must exist and it must be unique.*/
		std::string nodeName;
		unsigned int nodeIndex;
		/** The position keys of this animation channel. Positions are
		 * specified as 3D vector. The array is mNumPositionKeys in size.
		 * If there are position keys, there will also be at least one
		 * scaling and one rotation key.*/
		std::vector<PositionKey> positionKeys;
		/** The rotation keys of this animation channel. Rotations are
		 *  given as quaternions,  which are 4D vectors. The array is
		 *  mNumRotationKeys in size.
		 * If there are rotation keys, there will also be at least one
		 * scaling and one position key. */
		std::vector<RotationKey> rotationKeys;
		/** The scaling keys of this animation channel. Scalings are
		 *  specified as 3D vector. The array is mNumScalingKeys in size.
		 * If there are scaling keys, there will also be at least one
		 * position and one rotation key.*/
		std::vector<ScalingKey> scalingKeys;
		/** Defines how the animation behaves before the first
		 *  key is encountered.
		 *  The default value is aiAnimBehaviour_DEFAULT (the original
		 *  transformation matrix of the affected node is used).*/
		aiAnimBehaviour preState;
		/** Defines how the animation behaves after the last
		 *  key was processed.
		 *  The default value is aiAnimBehaviour_DEFAULT (the original
		 *  transformation matrix of the affected node is taken).*/
		aiAnimBehaviour postState;
	};

	struct MeshAnimationChannel
	{
		struct MeshKey
		{
			/** The time of this key */
			float time;
			/** Index into the aiMesh::mAnimMeshes array of the
			 *  mesh corresponding to the #aiMeshAnim hosting this
			 *  key frame. The referenced anim mesh is evaluated
			 *  according to the rules defined in the docs for #aiAnimMesh.*/
			unsigned int value;
		};

		/** Name of the mesh to be animated. An empty string is not allowed,
		 *  animated meshes need to be named (not necessarily uniquely,
		 *  the name can basically serve as wild-card to select a group
		 *  of meshes with similar animation setup)*/
		std::string name;
		unsigned int meshIndex;
		/** Key frames of the animation. May not be NULL. */
		std::vector<MeshKey> keys;
	};

	struct MeshMorphAnimation
	{
		struct MeshMorphKey
		{
			/** The time of this key */
			float time;
			/** The values and weights at the time of this key */
			std::vector<uint> values;
			/** The number of values and weights */
			std::vector<float> weights;

			[[nodiscard]] uint getNumberOfValuesAndWeights() const
			{
				assert(values.size() == weights.size());
				return static_cast<uint>(values.size());
			}
		};

		/** Name of the mesh to be animated. An empty string is not allowed,
		 *  animated meshes need to be named (not necessarily uniquely,
		 *  the name can basically serve as wildcard to select a group
		 *  of meshes with similar animation setup)*/
		std::string name;
		unsigned int meshIndex;
		// Key frames of the animation
		std::vector<MeshMorphKey> keys;
	};

	struct MeshAnimation
	{
		std::string name;
		double duration;
		double ticksPerSecond;

		std::vector<AnimationChannel> channels;
		std::vector<MeshAnimationChannel> meshChannels;
		std::vector<MeshMorphAnimation> morphChannels;
	};

	struct AssimpNode
	{
		void update(std::vector<AssimpNode> &nodes, const glm::mat4 &T);

		SIMDMat4 localTransform = glm::mat4(1.0f);
		SIMDMat4 combinedTransform = glm::mat4(1.0f);

		std::vector<uint> children;
		std::vector<uint> meshes;
	};

  public:
	rfw::SceneObject object;

	struct MeshBoneWeight
	{
		unsigned int vertexId;
		float weight;
	};

	struct MeshBone
	{
		std::string name;
		std::string nodeName;
		unsigned int nodeIndex;
		std::vector<int> vertexIDs;
		std::vector<float> weights;
		SIMDMat4 offsetMatrix;
	};

	struct MeshInfo
	{
		uint vertexOffset;
		uint vertexCount;
		uint faceOffset;
		uint faceCount;
		uint materialIdx;
		uint nodeIndex;

		std::vector<MeshBone> bones;
		std::vector<uvec4> joints;
		std::vector<vec4> weights;

		bool dirty = true;
	};

	AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
	AssimpObject(std::string_view filename, MaterialList *matList, uint ID, const glm::mat4 &matrix = glm::identity<glm::mat4>(), bool normalize = false,
				 int material = -1);
	~AssimpObject() override = default;

	void transformTo(float timeInSeconds) override;
	void updateTriangles();
	void updateTriangles(const std::vector<glm::vec2> &uvs);

	size_t traverseNode(const aiNode *node, int parentIdx, std::vector<AssimpNode> *storage, std::map<std::string, uint> *nodeNameMapping);

	Triangle *getTriangles() override { return m_Triangles.data(); }
	glm::vec4 *getVertices() override { return m_CurrentVertices.data(); }

	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const override;
	[[nodiscard]] const std::vector<SIMDMat4> &getMeshTransforms() const override;
	[[nodiscard]] std::vector<bool> getChangedMeshes() override;
	[[nodiscard]] std::vector<bool> getChangedMeshMatrices() override;

	[[nodiscard]] bool isAnimated() const override { return !m_Animations.empty(); }
	[[nodiscard]] const std::vector<std::vector<int>> &getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize) override;

  protected:
	void prepareMeshes(RenderSystem &rs) override;

  private:
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<std::pair<size_t, rfw::Mesh>> m_RfwMeshes;
	std::vector<SIMDMat4> m_MeshTransforms;
	std::vector<bool> m_ChangedMeshTransforms;

	std::vector<size_t> m_NodesWithMeshes;
	std::vector<AssimpNode> m_SceneGraph;
	std::map<std::string, uint> m_NodeNameMapping;

	std::vector<std::vector<uint>> m_MeshMapping;
	std::vector<MeshInfo> m_Meshes;

	// Up to date data if we have animations
	std::vector<Triangle> m_Triangles;
	std::vector<glm::vec4> m_CurrentVertices;
	std::vector<glm::vec3> m_CurrentNormals;

	// Original scene data
	std::vector<glm::uvec3> m_Indices;
	std::vector<uint> m_MaterialIndices;
	std::vector<glm::vec4> m_BaseVertices;
	std::vector<glm::vec3> m_BaseNormals;
	std::vector<glm::vec2> m_TexCoords;

	std::vector<MeshAnimation> m_Animations;

	std::string m_File;
	int m_ID = -1;
	bool m_IsAnimated = false, m_HasUpdated = false;
};
} // namespace rfw