#pragma once

#include <rfw/math.h>

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

#include <rfw/geometry/triangles.h>
#include <rfw/context/structs.h>
#include <rfw/material_list.h>
#include <rfw/geometry/triangles.h>
#include <rfw/geometry/gltf/hierarcy.h>

#include <rfw/geometry/gltf/object.h>
#include <rfw/geometry/gltf/skinning.h>

namespace rfw::geometry::assimp
{
class Object : public SceneTriangles
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

		rfw::simd::matrix4 localTransform = glm::mat4(1.0f);
		rfw::simd::matrix4 combinedTransform = glm::mat4(1.0f);

		std::vector<uint> children;
		std::vector<uint> meshes;
	};

  public:
	gltf::SceneObject object;

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
		rfw::simd::matrix4 offsetMatrix;
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

	Object(std::string_view filename, material_list *matList, uint ID,
		   const glm::mat4 &matrix = glm::identity<glm::mat4>(), int material = -1);
	Object(std::string_view filename, material_list *matList, uint ID,
		   const glm::mat4 &matrix = glm::identity<glm::mat4>(), bool normalize = false, int material = -1);
	~Object() override = default;

	void set_time(float timeInSeconds) override;
	void updateTriangles();
	void updateTriangles(const std::vector<glm::vec2> &uvs);

	size_t traverseNode(const aiNode *node, int parentIdx, std::vector<AssimpNode> *storage,
						std::map<std::string, uint> *nodeNameMapping);

	Triangle *get_triangles() override { return m_Triangles.data(); }
	glm::vec4 *get_vertices() override { return m_CurrentVertices.data(); }

	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &get_meshes() const override;
	[[nodiscard]] const std::vector<rfw::simd::matrix4> &get_mesh_matrices() const override;
	[[nodiscard]] std::vector<bool> get_changed_meshes() override;
	[[nodiscard]] std::vector<bool> get_changed_matrices() override;

	[[nodiscard]] bool is_animated() const override { return !m_Animations.empty(); }
	[[nodiscard]] const std::vector<std::vector<int>> &get_light_indices(const std::vector<bool> &matLightFlags,
																		 bool reinitialize) override;

  protected:
	void prepare_meshes(system &rs) override;

  private:
	std::vector<std::vector<int>> m_LightIndices;
	std::vector<std::pair<size_t, rfw::Mesh>> m_RfwMeshes;
	std::vector<rfw::simd::matrix4> m_MeshTransforms;
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
	std::vector<simd::vector4> m_BaseVertices;
	std::vector<simd::vector4> m_BaseNormals;
	std::vector<glm::vec2> m_TexCoords;

	std::vector<MeshAnimation> m_Animations;

	std::string m_File;
	int m_ID = -1;
	bool m_IsAnimated = false;
	bool m_HasUpdated = false;
};
} // namespace rfw::geometry::asimp