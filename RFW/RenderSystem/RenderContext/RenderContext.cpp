#include "Structures.h"

namespace rfw
{

float Triangle::calculateArea(const glm::vec3 &v0, const glm::vec3 &v1, const glm::vec3 &v2)
{
	const float a = length(v1 - v0);
	const float b = length(v2 - v1);
	const float c = length(v0 - v2);
	const float s = (a + b + c) * 0.5f;
	return sqrtf(s * (s - a) * (s - b) * (s - c)); // Heron's formula
}

void Triangle::updateArea() { area = calculateArea(vertex0, vertex1, vertex2); }
} // namespace rfw