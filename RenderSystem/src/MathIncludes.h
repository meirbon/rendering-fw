#pragma once

#ifdef _WIN32
#define NOMINMAX
#endif

#define GLM_FORCE_AVX2

#include <cmath>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/simd/geometric.h>
#include <glm/gtx/matrix_major_storage.hpp>
#include <immintrin.h>

using namespace glm;

namespace rfw
{
namespace simd
{
struct matrix4
{
	matrix4() = default;
	matrix4(glm::mat4 m) : matrix(m) {}

	union {
		glm::mat4 matrix;
#if GLM_ARCH & GLM_ARCH_SSE2_BIT
		glm_vec4 cols[4];
#else
		glm::vec4 cols[4];
#endif
	};

	matrix4 operator*(float op) const;
	matrix4 operator/(float op) const;
	matrix4 operator+(float op) const;
	matrix4 operator-(float op) const;
	matrix4 operator*(const matrix4 &op) const;
	matrix4 operator+(const matrix4 &op) const;
	void operator*=(float op);
	void operator/=(float op);
	void operator+=(float op);
	void operator-=(float op);
	void operator*=(const matrix4 &op);
	void operator+=(const matrix4 &op);
	matrix4 inversed() const;
	matrix4 transposed() const;
};

struct vector4
{
	union {
		glm::vec4 vec;
		__m128 vec_4;
	};

	static vector4 zero();

	vector4() = default;
	vector4(const float *a);
	vector4(const float a) : vec(a) {}
	vector4(const float a, const float b, const float c, const float d) : vec(a, b, c, d) {}
	vector4(const float *a, const simd::vector4 &mask);
	vector4(const glm::vec2 &v1, const glm::vec2 &v2);
	vector4(const glm::vec3 &v, const float w);
	vector4(const glm::vec3 &v);
	vector4(const glm::vec4 &v);
	vector4(const __m128 &a4) : vec_4(a4) {}
	vector4(const __m128i &a4) : vec_4(_mm_castsi128_ps(a4)) {}

	explicit operator glm::vec4 &() { return vec; }
	explicit operator const glm::vec4 &() const { return vec; }
	explicit operator __m128() const { return vec_4; }

	[[nodiscard]] float x() const { return vec.x; }
	[[nodiscard]] float y() const { return vec.x; }
	[[nodiscard]] float z() const { return vec.x; }
	[[nodiscard]] float w() const { return vec.x; }

	[[nodiscard]] float dot(const vector4 &op) const;
	[[nodiscard]] float length() const;
	[[nodiscard]] vector4 min(const vector4 &op) const;
	[[nodiscard]] vector4 max(const vector4 &op) const;
	[[nodiscard]] vector4 abs() const;

	vector4 operator*(const vector4 &op) const;
	vector4 operator/(const vector4 &op) const;
	vector4 operator+(const vector4 &op) const;
	vector4 operator-(const vector4 &op) const;
	vector4 operator>(const vector4 &op) const;
	vector4 operator<(const vector4 &op) const;
	vector4 operator>=(const vector4 &op) const;
	vector4 operator<=(const vector4 &op) const;
	vector4 operator==(const vector4 &op) const;
	vector4 operator!=(const vector4 &op) const;
	vector4 operator&(const vector4 &op) const;
	vector4 operator|(const vector4 &op) const;

	[[nodiscard]] glm::bvec4 to_mask() const;
	[[nodiscard]] int move_mask() const;

	void write_to(float *loc);
	void write_to(float *loc, const vector4 &mask);
	void store(const vector4 &result, const vector4 &mask);
};

glm::vec4 operator*(const glm::vec4 &op, const matrix4 &mat);
glm::vec4 operator*(const matrix4 &mat, const glm::vec4 &op);

vector4 operator*(const vector4 &op, const matrix4 &mat);
vector4 operator*(const matrix4 &mat, const vector4 &op);

vector4 operator*(const vector4 &op1, float op2);
vector4 operator/(const vector4 &op1, float op2);
vector4 operator+(const vector4 &op1, float op2);
vector4 operator-(const vector4 &op1, float op2);
vector4 operator>(const vector4 &op1, float op2);
vector4 operator<(const vector4 &op1, float op2);
vector4 operator>=(const vector4 &op1, float op2);
vector4 operator<=(const vector4 &op1, float op2);
vector4 operator==(const vector4 &op1, float op2);
vector4 operator!=(const vector4 &op1, float op2);
vector4 operator&(const vector4 &op1, float op2);
vector4 operator|(const vector4 &op1, float op2);

vector4 operator*(float op1, const vector4 &op2);
vector4 operator/(float op1, const vector4 &op2);
vector4 operator+(float op1, const vector4 &op2);
vector4 operator-(float op1, const vector4 &op2);
vector4 operator>(float op1, const vector4 &op2);
vector4 operator<(float op1, const vector4 &op2);
vector4 operator>=(float op1, const vector4 &op2);
vector4 operator<=(float op1, const vector4 &op2);
vector4 operator==(float op1, const vector4 &op2);
vector4 operator!=(float op1, const vector4 &op2);
vector4 operator&(float op1, const vector4 &op2);
vector4 operator|(float op1, const vector4 &op2);

__m128 log_ps(__m128 x);
__m128 exp_ps(__m128 x);
__m128 sin_ps(__m128 x);
__m128 cos_ps(__m128 x);
void sincos_ps(__m128 x, __m128 *s, __m128 *c);
__m128 atan2_ps(__m128 x, __m128 y);
__m128 acos_ps(__m128 x);
} // namespace simd
} // namespace rfw