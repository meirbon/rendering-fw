#pragma once

#include <vector>
#include <string>

#include <MathIncludes.h>

#include <glm/gtx/matrix_major_storage.hpp>

#define ROW_MAJOR_MESH_SKIN 1

namespace rfw
{
class MeshSkin
{
  public:
	std::string name;
	std::vector<int> jointNodes;

	std::vector<glm::mat4> inverseBindMatrices;
	std::vector<glm::mat4> jointMatrices;
};

} // namespace rfw