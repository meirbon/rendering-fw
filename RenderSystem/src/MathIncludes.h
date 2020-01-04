#pragma once

#define NOMINMAX

#define GLM_FORCE_AVX2

#include <cmath>
#include <glm/glm.hpp>
#include <glm/ext.hpp>

using namespace glm;

struct SIMDMat4
{
	SIMDMat4() = default;
	SIMDMat4(glm::mat4 m) : matrix(m) {}

	union {
		glm::mat4 matrix;
#if GLM_ARCH & GLM_ARCH_SSE2_BIT
		glm_vec4 cols[4];
#else
		glm::vec4 cols[4];
#endif
	};

	SIMDMat4 operator*(float op) const;
	SIMDMat4 operator/(float op) const;
	SIMDMat4 operator+(float op) const;
	SIMDMat4 operator-(float op) const;
	SIMDMat4 operator*(const SIMDMat4 &op) const;
	SIMDMat4 operator+(const SIMDMat4 &op) const;
	SIMDMat4 inversed() const;
	SIMDMat4 transposed() const;
};
