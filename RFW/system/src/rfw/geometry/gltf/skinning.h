#pragma once

#include <vector>
#include <string>
#include <array>

#include <rfw/math.h>

#define ROW_MAJOR_MESH_SKIN 0

namespace rfw::geometry::gltf
{

class MeshSkin
{
  public:
	std::string name;
	std::vector<int> jointNodes;

	std::vector<simd::matrix4> inverseBindMatrices;
	std::vector<simd::matrix4> jointMatrices;
};

class MeshBone
{
  public:
	std::string name;
	unsigned int nodeIndex;

	std::vector<unsigned short> vertexIDs;
	std::vector<float> vertexWeights;
	glm::mat4 offsetMatrix;
};

} // namespace rfw::geometry::gltf