#pragma once

#include <vector>
#include <string>

#include <MathIncludes.h>

#include <glm/gtx/matrix_major_storage.hpp>

namespace rfw
{
class MeshSkin
{
  public:
	std::string name;

	std::vector<int> joints;
	std::vector<glm::mat4> inverseBindMatrices;
	std::vector<glm::mat4> jointMatrices;
};

} // namespace rfw