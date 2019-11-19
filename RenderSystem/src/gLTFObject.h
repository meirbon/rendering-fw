#pragma once

#include <MathIncludes.h>

#include <string>
#include <string_view>

#include <Structures.h>
#include "MaterialList.h"
#include "SceneTriangles.h"

#include "gLTFAnimation.h"

namespace rfw
{

class gTLFObject : public SceneTriangles
{
  public:
	struct gLTFNode
	{
		bool update(gTLFObject &object, glm::mat4 &T);
		void updateTransform();
		void prepareLights();
		void updateLights();

		std::string name;
		glm::mat4 combinedTransform; // Combined transform of parent nodes
		glm::mat4 localTransform;	// T * R * S

		glm::vec3 translation;
		glm::quat rotation;
		glm::vec3 scale;

		glm::mat4 matrix;

		int ID = -1;
		int meshID = -1;
		int skinID = -1;

		std::vector<float> weights;
		bool hasLTris = false;
		bool morphed = false;
		bool transformed = false;
		bool treeChanged = false;
		bool wasModified = false;

		std::vector<int> childIndices;
	};

	struct gLTFPose
	{
		std::vector<glm::vec3> positions;
		std::vector<glm::vec3> normals;
		std::vector<glm::vec3> tangents;
	};

	struct gLTFSkin
	{
		std::string name;
		int skeletonRoot = 0;

		std::vector<glm::mat4> inverseBindMatrices;
		std::vector<glm::mat4> jointMatrices;
		std::vector<int> joints;
	};

	struct gLTFMesh
	{
		void setPose(gTLFObject &object, const gLTFSkin &skin);
		void setPose(gTLFObject &object, const std::vector<float> &weights);

		unsigned int materialIdx;

		unsigned int vertexOffset;
		unsigned int vertexCount;

		unsigned int poseOffset;
		unsigned int poseCount;

		unsigned int jointOffset;
		unsigned int jointCount;

		unsigned int weightOffset;
		unsigned int weightCount;
	};

	explicit gTLFObject(std::string_view filename, MaterialList *matList, uint ID,
						const glm::mat4 &matrix = glm::identity<glm::mat4>(), bool normalize = false,
						int material = -1);
	~gTLFObject() = default;

	void transformTo(float timeInSeconds = 0.0f) override;

	Triangle *getTriangles() override;
	glm::vec4 *getVertices() override;

	[[nodiscard]] rfw::Mesh getMesh() const override;

	bool isAnimated() const override;
	uint getAnimationCount() const override;
	void setAnimation(uint index) override;
	uint getMaterialForPrim(uint primitiveIdx) const override;

	std::vector<uint> getLightIndices(const std::vector<bool> &matLightFlags) const override;

	const std::string file;

	std::vector<gLTFPose> m_Poses;
	std::vector<gLTFMesh> m_Meshes;
	std::vector<gLTFNode> m_Nodes;
	std::vector<gLTFSkin> m_Skins;
	std::vector<glm::uvec4> m_Joints;
	std::vector<glm::vec4> m_Weights;

  private:
	void build(const std::vector<int> &indices, const std::vector<glm::vec3> &vertices,
			   const std::vector<glm::vec3> &normals, const std::vector<glm::vec2> &uvs,
			   const std::vector<gLTFPose> &poses, const std::vector<glm::uvec4> &joints,
			   const std::vector<glm::vec4> &weights, int materialIdx);

	std::vector<gLTFAnimation> m_Animations;

	// Up to date data if we have animations
	std::vector<Triangle> m_Triangles;
	std::vector<glm::vec4> m_CurrentVertices;
	std::vector<glm::vec3> m_CurrentNormals;

	// Original scene data
	// std::vector<glm::uvec3> m_Indices;
	std::vector<uint> m_MaterialIndices;
	std::vector<glm::vec4> m_BaseVertices;
	std::vector<glm::vec3> m_BaseNormals;
	std::vector<glm::vec2> m_BaseTexCoords;

	unsigned int m_BaseMaterialIdx;
	int m_ID = -1;
	bool m_IsAnimated = false, m_HasUpdated = false;
};

} // namespace rfw