#pragma once

#include <glm/glm.hpp>
#include "Ray.h"
#include <rfw.h>

#define EPSILON_TRIANGLE 0.0001f

namespace rfw::triangle
{

bool intersect(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt, const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2,
			   float epsilon = EPSILON_TRIANGLE);
bool intersect(const glm::vec3 &org, const glm::vec3 &dir, float tmin, float *rayt, const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2,
			   float epsilon = EPSILON_TRIANGLE);
bool intersect_opt(const glm::vec3 org, const glm::vec3 dir, float tmin, float *rayt, const glm::vec3 p0, const glm::vec3 e1, const glm::vec3 e2,
				   float epsilon = EPSILON_TRIANGLE);

glm::vec3 getBaryCoords(const glm::vec3 &p, const glm::vec3 &normal, const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);
glm::vec3 getBaryCoords(const glm::vec3 &p, const glm::vec3 &normal, const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2);

rfw::bvh::AABB getBounds(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);
rfw::bvh::AABB getBounds(const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2);

glm::vec3 getRandomPointOnSurface(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2, float r1, float r2);
glm::vec3 getRandomPointOnSurface(const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2, float r1, float r2);

glm::vec3 getFaceNormal(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);
glm::vec3 getFaceNormal(const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2);

glm::vec3 getNormal(const glm::vec3 &bary, const glm::vec3 &n0, const glm::vec3 &n1, const glm::vec3 &n2);

glm::vec2 getTexCoords(const glm::vec3 &bary, const glm::vec2 &t0, const glm::vec2 &t1, const glm::vec2 &t2);

float getArea(const glm::vec3 &p0, const glm::vec3 &p1, const glm::vec3 &p2);
float getArea(const glm::vec4 &p0, const glm::vec4 &p1, const glm::vec4 &p2);

} // namespace rfw::triangle