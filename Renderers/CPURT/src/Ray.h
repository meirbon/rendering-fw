#pragma once

#include <glm/glm.hpp>

struct Ray
{
	glm::vec3 origin = glm::vec3(0.0f);
	float t = 1e34f;

	glm::vec3 direction = glm::vec3(0, 0, 1);
	int primIdx = -1;

	[[nodiscard]] bool isValid() const { return primIdx >= 0; }

	void reset()
	{
		t = 1e34f;
		primIdx = -1;
	}
};
