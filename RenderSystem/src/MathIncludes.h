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

struct SIMDVec4
{
	union {
		glm::vec4 vec;
		__m128 vec_4;
	};

	static SIMDVec4 zero();

	SIMDVec4() = default;
	SIMDVec4(const float *a);
	SIMDVec4(const float a) : vec(a) {}
	SIMDVec4(const float a, const float b, const float c, const float d) : vec(a, b, c, d) {}
	SIMDVec4(const glm::vec2 &v1, const glm::vec2 &v2);
	SIMDVec4(const glm::vec3 &v, const float w);
	SIMDVec4(const glm::vec3 &v);
	SIMDVec4(const glm::vec4 &v);
	SIMDVec4(const __m128 &a4) : vec_4(a4) {}

	explicit operator glm::vec4 &() { return vec; }
	explicit operator const glm::vec4 &() const { return vec; }
	explicit operator __m128() const { return vec_4; }

	[[nodiscard]] float x() const { return vec.x; }
	[[nodiscard]] float y() const { return vec.x; }
	[[nodiscard]] float z() const { return vec.x; }
	[[nodiscard]] float w() const { return vec.x; }

	[[nodiscard]] float dot(const SIMDVec4 &op) const;
	[[nodiscard]] SIMDVec4 min(const SIMDVec4 &op) const;
	[[nodiscard]] SIMDVec4 max(const SIMDVec4 &op) const;
	[[nodiscard]] SIMDVec4 abs() const;

	SIMDVec4 operator*(const SIMDVec4 &op) const;
	SIMDVec4 operator/(const SIMDVec4 &op) const;
	SIMDVec4 operator+(const SIMDVec4 &op) const;
	SIMDVec4 operator-(const SIMDVec4 &op) const;
	SIMDVec4 operator>(const SIMDVec4 &op) const;
	SIMDVec4 operator<(const SIMDVec4 &op) const;
	SIMDVec4 operator>=(const SIMDVec4 &op) const;
	SIMDVec4 operator<=(const SIMDVec4 &op) const;
	SIMDVec4 operator==(const SIMDVec4 &op) const;
	SIMDVec4 operator!=(const SIMDVec4 &op) const;
	SIMDVec4 operator&(const SIMDVec4 &op) const;
	SIMDVec4 operator|(const SIMDVec4 &op) const;

	[[nodiscard]] glm::bvec4 to_mask() const;
	[[nodiscard]] int move_mask() const;

	void store(float *loc);
	void store(float *loc, const SIMDVec4 &mask);
	void store_if(const SIMDVec4 &result, const SIMDVec4 &mask);
};

glm::vec4 operator*(const glm::vec4 &op, const SIMDMat4 &mat);
glm::vec4 operator*(const SIMDMat4 &mat, const glm::vec4 &op);

SIMDVec4 operator*(const SIMDVec4 &op, const SIMDMat4 &mat);
SIMDVec4 operator*(const SIMDMat4 &mat, const SIMDVec4 &op);

SIMDVec4 operator*(const SIMDVec4 &op1, float op2);
SIMDVec4 operator/(const SIMDVec4 &op1, float op2);
SIMDVec4 operator+(const SIMDVec4 &op1, float op2);
SIMDVec4 operator-(const SIMDVec4 &op1, float op2);
SIMDVec4 operator>(const SIMDVec4 &op1, float op2);
SIMDVec4 operator<(const SIMDVec4 &op1, float op2);
SIMDVec4 operator>=(const SIMDVec4 &op1, float op2);
SIMDVec4 operator<=(const SIMDVec4 &op1, float op2);
SIMDVec4 operator==(const SIMDVec4 &op1, float op2);
SIMDVec4 operator!=(const SIMDVec4 &op1, float op2);
SIMDVec4 operator&(const SIMDVec4 &op1, float op2);
SIMDVec4 operator|(const SIMDVec4 &op1, float op2);

SIMDVec4 operator*(float op1, const SIMDVec4 &op2);
SIMDVec4 operator/(float op1, const SIMDVec4 &op2);
SIMDVec4 operator+(float op1, const SIMDVec4 &op2);
SIMDVec4 operator-(float op1, const SIMDVec4 &op2);
SIMDVec4 operator>(float op1, const SIMDVec4 &op2);
SIMDVec4 operator<(float op1, const SIMDVec4 &op2);
SIMDVec4 operator>=(float op1, const SIMDVec4 &op2);
SIMDVec4 operator<=(float op1, const SIMDVec4 &op2);
SIMDVec4 operator==(float op1, const SIMDVec4 &op2);
SIMDVec4 operator!=(float op1, const SIMDVec4 &op2);
SIMDVec4 operator&(float op1, const SIMDVec4 &op2);
SIMDVec4 operator|(float op1, const SIMDVec4 &op2);

__m128 log_ps(__m128 x);
__m128 exp_ps(__m128 x);
__m128 sin_ps(__m128 x);
__m128 cos_ps(__m128 x);
void sincos_ps(__m128 x, __m128 *s, __m128 *c);
__m128 atan2_ps(__m128 x, __m128 y);
__m128 acos_ps(__m128 x);