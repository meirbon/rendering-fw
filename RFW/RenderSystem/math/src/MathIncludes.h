#pragma once

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX 1
#endif

#undef min
#undef max
#endif

#include <cmath>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <immintrin.h>

using namespace glm;

namespace rfw
{
namespace simd
{
#ifdef _MSC_VER /* visual c++ */
#define ALIGN16_BEG __declspec(align(16))
#define ALIGN16_END
#else /* gcc or icc */
#define ALIGN16_BEG
#define ALIGN16_END __attribute__((aligned(16)))
#endif

#define _PS_CONST(Name, Val) static const ALIGN16_BEG float _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PI32_CONST(Name, Val) static const ALIGN16_BEG int _pi32_##Name[4] ALIGN16_END = {Val, Val, Val, Val}
#define _PS_CONST_TYPE(Name, Type, Val) static const ALIGN16_BEG Type _ps_##Name[4] ALIGN16_END = {Val, Val, Val, Val}

_PS_CONST(1, 1.0f);
_PS_CONST(0p5, 0.5f);
/* the smallest non denormalized float number */
_PS_CONST_TYPE(min_norm_pos, int, 0x00800000);
_PS_CONST_TYPE(mant_mask, int, 0x7f800000);
_PS_CONST_TYPE(inv_mant_mask, int, ~0x7f800000);

_PS_CONST_TYPE(sign_mask, int, (int)0x80000000);
_PS_CONST_TYPE(inv_sign_mask, int, ~0x80000000);

_PI32_CONST(1, 1);
_PI32_CONST(inv1, ~1);
_PI32_CONST(2, 2);
_PI32_CONST(4, 4);
_PI32_CONST(0x7f, 0x7f);

_PS_CONST(cephes_SQRTHF, 0.707106781186547524f);
_PS_CONST(cephes_log_p0, 7.0376836292E-2f);
_PS_CONST(cephes_log_p1, -1.1514610310E-1f);
_PS_CONST(cephes_log_p2, 1.1676998740E-1f);
_PS_CONST(cephes_log_p3, -1.2420140846E-1f);
_PS_CONST(cephes_log_p4, +1.4249322787E-1f);
_PS_CONST(cephes_log_p5, -1.6668057665E-1f);
_PS_CONST(cephes_log_p6, +2.0000714765E-1f);
_PS_CONST(cephes_log_p7, -2.4999993993E-1f);
_PS_CONST(cephes_log_p8, +3.3333331174E-1f);
_PS_CONST(cephes_log_q1, -2.12194440e-4f);
_PS_CONST(cephes_log_q2, 0.693359375f);

_PS_CONST(exp_hi, 88.3762626647949f);
_PS_CONST(exp_lo, -88.3762626647949f);

_PS_CONST(cephes_LOG2EF, 1.44269504088896341f);
_PS_CONST(cephes_exp_C1, 0.693359375f);
_PS_CONST(cephes_exp_C2, -2.12194440e-4f);

_PS_CONST(cephes_exp_p0, 1.9875691500E-4f);
_PS_CONST(cephes_exp_p1, 1.3981999507E-3f);
_PS_CONST(cephes_exp_p2, 8.3334519073E-3f);
_PS_CONST(cephes_exp_p3, 4.1665795894E-2f);
_PS_CONST(cephes_exp_p4, 1.6666665459E-1f);
_PS_CONST(cephes_exp_p5, 5.0000001201E-1f);

_PS_CONST(minus_cephes_DP1, -0.78515625f);
_PS_CONST(minus_cephes_DP2, -2.4187564849853515625e-4f);
_PS_CONST(minus_cephes_DP3, -3.77489497744594108e-8f);
_PS_CONST(sincof_p0, -1.9515295891E-4f);
_PS_CONST(sincof_p1, 8.3321608736E-3f);
_PS_CONST(sincof_p2, -1.6666654611E-1f);
_PS_CONST(coscof_p0, 2.443315711809948E-005f);
_PS_CONST(coscof_p1, -1.388731625493765E-003f);
_PS_CONST(coscof_p2, 4.166664568298827E-002f);
_PS_CONST(cephes_FOPI, 1.27323954473516f); // 4 / M_PI

struct matrix4
{
	matrix4() = default;
	matrix4(glm::mat4 m) : matrix(m) {}

	union {
		glm::mat4 matrix;
		union {
			__m128 cols[4];
			__m256 cols8[2];
		};
		glm::vec4 cols_vecs[4];
	};

	__m128 &operator[](int index) { return cols[index]; }
	const __m128 &operator[](int index) const { return cols[index]; }

	matrix4 operator*(float op) const
	{
		matrix4 result;
#if 1
		const __m256 op8 = _mm256_set1_ps(op);

		result.cols8[0] = _mm256_mul_ps(cols8[0], op8);
		result.cols8[1] = _mm256_mul_ps(cols8[1], op8);
#else
		const __m128 op4 = _mm_set1_ps(op);

		result.cols[0] = _mm_mul_ps(cols[0], op4);
		result.cols[1] = _mm_mul_ps(cols[1], op4);
		result.cols[2] = _mm_mul_ps(cols[2], op4);
		result.cols[3] = _mm_mul_ps(cols[3], op4);
#endif
		return result;
	}
	matrix4 operator/(float op) const
	{
		matrix4 result;
#if 1
		const __m256 op8 = _mm256_set1_ps(op);

		result.cols8[0] = _mm256_div_ps(cols8[0], op8);
		result.cols8[1] = _mm256_div_ps(cols8[1], op8);
#else
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_div_ps(cols[0], op4);
		result.cols[1] = _mm_div_ps(cols[1], op4);
		result.cols[2] = _mm_div_ps(cols[2], op4);
		result.cols[3] = _mm_div_ps(cols[3], op4);
#endif
		return result;
	}
	matrix4 operator+(float op) const
	{
		matrix4 result;
#if 1
		const __m256 op8 = _mm256_set1_ps(op);

		result.cols8[0] = _mm256_add_ps(cols8[0], op8);
		result.cols8[1] = _mm256_add_ps(cols8[1], op8);
#else
		const __m128 op4 = _mm_set1_ps(op);

		result.cols[0] = _mm_add_ps(cols[0], op4);
		result.cols[1] = _mm_add_ps(cols[1], op4);
		result.cols[2] = _mm_add_ps(cols[2], op4);
		result.cols[3] = _mm_add_ps(cols[3], op4);
#endif
		return result;
	}
	matrix4 operator-(float op) const
	{
		matrix4 result;
#if 1
		const __m256 op8 = _mm256_set1_ps(op);

		result.cols8[0] = _mm256_sub_ps(cols8[0], op8);
		result.cols8[1] = _mm256_sub_ps(cols8[1], op8);
#else
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_sub_ps(cols[0], op4);
		result.cols[1] = _mm_sub_ps(cols[1], op4);
		result.cols[2] = _mm_sub_ps(cols[2], op4);
		result.cols[3] = _mm_sub_ps(cols[3], op4);
#endif
		return result;
	}
	matrix4 operator*(const matrix4 &op) const
	{
		matrix4 result;

		{
			__m128 e0 = _mm_shuffle_ps(op[0], op[0], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 e1 = _mm_shuffle_ps(op[0], op[0], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 e2 = _mm_shuffle_ps(op[0], op[0], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 e3 = _mm_shuffle_ps(op[0], op[0], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 m0 = _mm_mul_ps(cols[0], e0);
			__m128 m1 = _mm_mul_ps(cols[1], e1);
			__m128 m2 = _mm_mul_ps(cols[2], e2);
			__m128 m3 = _mm_mul_ps(cols[3], e3);

			__m128 a0 = _mm_add_ps(m0, m1);
			__m128 a1 = _mm_add_ps(m2, m3);
			__m128 a2 = _mm_add_ps(a0, a1);

			result[0] = a2;
		}

		{
			__m128 e0 = _mm_shuffle_ps(op[1], op[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 e1 = _mm_shuffle_ps(op[1], op[1], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 e2 = _mm_shuffle_ps(op[1], op[1], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 e3 = _mm_shuffle_ps(op[1], op[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 m0 = _mm_mul_ps(cols[0], e0);
			__m128 m1 = _mm_mul_ps(cols[1], e1);
			__m128 m2 = _mm_mul_ps(cols[2], e2);
			__m128 m3 = _mm_mul_ps(cols[3], e3);

			__m128 a0 = _mm_add_ps(m0, m1);
			__m128 a1 = _mm_add_ps(m2, m3);
			__m128 a2 = _mm_add_ps(a0, a1);

			result[1] = a2;
		}

		{
			__m128 e0 = _mm_shuffle_ps(op[2], op[2], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 e1 = _mm_shuffle_ps(op[2], op[2], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 e2 = _mm_shuffle_ps(op[2], op[2], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 e3 = _mm_shuffle_ps(op[2], op[2], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 m0 = _mm_mul_ps(cols[0], e0);
			__m128 m1 = _mm_mul_ps(cols[1], e1);
			__m128 m2 = _mm_mul_ps(cols[2], e2);
			__m128 m3 = _mm_mul_ps(cols[3], e3);

			__m128 a0 = _mm_add_ps(m0, m1);
			__m128 a1 = _mm_add_ps(m2, m3);
			__m128 a2 = _mm_add_ps(a0, a1);

			result[2] = a2;
		}

		{
			//(__m128&)_mm_shuffle_epi32(__m128i&)in2[0], _MM_SHUFFLE(3, 3, 3, 3))
			__m128 e0 = _mm_shuffle_ps(op[3], op[3], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 e1 = _mm_shuffle_ps(op[3], op[3], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 e2 = _mm_shuffle_ps(op[3], op[3], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 e3 = _mm_shuffle_ps(op[3], op[3], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 m0 = _mm_mul_ps(cols[0], e0);
			__m128 m1 = _mm_mul_ps(cols[1], e1);
			__m128 m2 = _mm_mul_ps(cols[2], e2);
			__m128 m3 = _mm_mul_ps(cols[3], e3);

			__m128 a0 = _mm_add_ps(m0, m1);
			__m128 a1 = _mm_add_ps(m2, m3);
			__m128 a2 = _mm_add_ps(a0, a1);

			result[3] = a2;
		}

		return result;
	}

	matrix4 operator+(const matrix4 &op) const
	{
		matrix4 out;
#if 1
		out.cols8[0] = _mm256_add_ps(cols8[0], op.cols8[0]);
		out.cols8[1] = _mm256_add_ps(cols8[1], op.cols8[1]);
#else
		out[0] = _mm_add_ps(cols[0], op[0]);
		out[1] = _mm_add_ps(cols[1], op[1]);
		out[2] = _mm_add_ps(cols[2], op[2]);
		out[3] = _mm_add_ps(cols[3], op[3]);
#endif
		return out;
	}

	matrix4 operator-(const matrix4 &op) const
	{
		matrix4 out;
#if 1
		out.cols8[0] = _mm256_sub_ps(cols8[0], op.cols8[0]);
		out.cols8[1] = _mm256_sub_ps(cols8[1], op.cols8[1]);
#else
		out[0] = _mm_sub_ps(cols[0], op[0]);
		out[1] = _mm_sub_ps(cols[1], op[1]);
		out[2] = _mm_sub_ps(cols[2], op[2]);
		out[3] = _mm_sub_ps(cols[3], op[3]);
#endif
		return out;
	}
	void operator*=(float op) { *this = (*this) * op; }
	void operator/=(float op) { *this = (*this) / op; }
	void operator+=(float op) { *this = (*this) + op; }
	void operator-=(float op) { *this = (*this) - op; }
	void operator*=(const matrix4 &op) { *this = (*this) * op; }
	void operator+=(const matrix4 &op) { *this = (*this) + op; }
	matrix4 inversed() const
	{
		__m128 Fac0;
		{
			//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
			//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
			//	valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
			//	valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac0 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac1;
		{
			//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
			//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
			//	valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
			//	valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac1 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac2;
		{
			//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
			//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
			//	valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
			//	valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac2 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac3;
		{
			//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
			//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
			//	valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
			//	valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac3 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac4;
		{
			//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
			//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
			//	valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
			//	valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac4 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac5;
		{
			//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
			//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
			//	valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
			//	valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac5 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 SignA = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
		__m128 SignB = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);

		// m[1][0]
		// m[0][0]
		// m[0][0]
		// m[0][0]
		__m128 Temp0 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][1]
		// m[0][1]
		// m[0][1]
		// m[0][1]
		__m128 Temp1 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][2]
		// m[0][2]
		// m[0][2]
		// m[0][2]
		__m128 Temp2 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][3]
		// m[0][3]
		// m[0][3]
		// m[0][3]
		__m128 Temp3 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

		// col0
		// + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
		// - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
		// + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
		// - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
		__m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
		__m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
		__m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
		__m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
		__m128 Add00 = _mm_add_ps(Sub00, Mul02);
		__m128 Inv0 = _mm_mul_ps(SignB, Add00);

		// col1
		// - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
		// + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
		// - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
		// + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
		__m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
		__m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
		__m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
		__m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
		__m128 Add01 = _mm_add_ps(Sub01, Mul05);
		__m128 Inv1 = _mm_mul_ps(SignA, Add01);

		// col2
		// + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
		// - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
		// + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
		// - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
		__m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
		__m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
		__m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
		__m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
		__m128 Add02 = _mm_add_ps(Sub02, Mul08);
		__m128 Inv2 = _mm_mul_ps(SignB, Add02);

		// col3
		// - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
		// + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
		// - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
		// + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
		__m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
		__m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
		__m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
		__m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
		__m128 Add03 = _mm_add_ps(Sub03, Mul11);
		__m128 Inv3 = _mm_mul_ps(SignA, Add03);

		__m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

		//	valType Determinant = m[0][0] * Inverse[0][0]
		//						+ m[0][1] * Inverse[1][0]
		//						+ m[0][2] * Inverse[2][0]
		//						+ m[0][3] * Inverse[3][0];
		__m128 Det0 = _mm_dp_ps(cols[0], Row2, 0xff);
		__m128 Rcp0 = _mm_div_ps(_mm_set1_ps(1.0f), Det0);
		//__m128 Rcp0 = _mm_rcp_ps(Det0);

		matrix4 out;

		//	Inverse /= Determinant;
		out[0] = _mm_mul_ps(Inv0, Rcp0);
		out[1] = _mm_mul_ps(Inv1, Rcp0);
		out[2] = _mm_mul_ps(Inv2, Rcp0);
		out[3] = _mm_mul_ps(Inv3, Rcp0);

		return out;
	}

	matrix4 inversed_lowp() const
	{
		__m128 Fac0;
		{
			//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
			//	valType SubFactor00 = m[2][2] * m[3][3] - m[3][2] * m[2][3];
			//	valType SubFactor06 = m[1][2] * m[3][3] - m[3][2] * m[1][3];
			//	valType SubFactor13 = m[1][2] * m[2][3] - m[2][2] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac0 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac1;
		{
			//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
			//	valType SubFactor01 = m[2][1] * m[3][3] - m[3][1] * m[2][3];
			//	valType SubFactor07 = m[1][1] * m[3][3] - m[3][1] * m[1][3];
			//	valType SubFactor14 = m[1][1] * m[2][3] - m[2][1] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac1 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac2;
		{
			//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
			//	valType SubFactor02 = m[2][1] * m[3][2] - m[3][1] * m[2][2];
			//	valType SubFactor08 = m[1][1] * m[3][2] - m[3][1] * m[1][2];
			//	valType SubFactor15 = m[1][1] * m[2][2] - m[2][1] * m[1][2];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac2 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac3;
		{
			//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
			//	valType SubFactor03 = m[2][0] * m[3][3] - m[3][0] * m[2][3];
			//	valType SubFactor09 = m[1][0] * m[3][3] - m[3][0] * m[1][3];
			//	valType SubFactor16 = m[1][0] * m[2][3] - m[2][0] * m[1][3];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(3, 3, 3, 3));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(3, 3, 3, 3));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac3 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac4;
		{
			//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
			//	valType SubFactor04 = m[2][0] * m[3][2] - m[3][0] * m[2][2];
			//	valType SubFactor10 = m[1][0] * m[3][2] - m[3][0] * m[1][2];
			//	valType SubFactor17 = m[1][0] * m[2][2] - m[2][0] * m[1][2];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(2, 2, 2, 2));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(2, 2, 2, 2));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac4 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 Fac5;
		{
			//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
			//	valType SubFactor05 = m[2][0] * m[3][1] - m[3][0] * m[2][1];
			//	valType SubFactor12 = m[1][0] * m[3][1] - m[3][0] * m[1][1];
			//	valType SubFactor18 = m[1][0] * m[2][1] - m[2][0] * m[1][1];

			__m128 Swp0a = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(1, 1, 1, 1));
			__m128 Swp0b = _mm_shuffle_ps(cols[3], cols[2], _MM_SHUFFLE(0, 0, 0, 0));

			__m128 Swp00 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(0, 0, 0, 0));
			__m128 Swp01 = _mm_shuffle_ps(Swp0a, Swp0a, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp02 = _mm_shuffle_ps(Swp0b, Swp0b, _MM_SHUFFLE(2, 0, 0, 0));
			__m128 Swp03 = _mm_shuffle_ps(cols[2], cols[1], _MM_SHUFFLE(1, 1, 1, 1));

			__m128 Mul00 = _mm_mul_ps(Swp00, Swp01);
			__m128 Mul01 = _mm_mul_ps(Swp02, Swp03);
			Fac5 = _mm_sub_ps(Mul00, Mul01);
		}

		__m128 SignA = _mm_set_ps(1.0f, -1.0f, 1.0f, -1.0f);
		__m128 SignB = _mm_set_ps(-1.0f, 1.0f, -1.0f, 1.0f);

		// m[1][0]
		// m[0][0]
		// m[0][0]
		// m[0][0]
		__m128 Temp0 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Vec0 = _mm_shuffle_ps(Temp0, Temp0, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][1]
		// m[0][1]
		// m[0][1]
		// m[0][1]
		__m128 Temp1 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(1, 1, 1, 1));
		__m128 Vec1 = _mm_shuffle_ps(Temp1, Temp1, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][2]
		// m[0][2]
		// m[0][2]
		// m[0][2]
		__m128 Temp2 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(2, 2, 2, 2));
		__m128 Vec2 = _mm_shuffle_ps(Temp2, Temp2, _MM_SHUFFLE(2, 2, 2, 0));

		// m[1][3]
		// m[0][3]
		// m[0][3]
		// m[0][3]
		__m128 Temp3 = _mm_shuffle_ps(cols[1], cols[0], _MM_SHUFFLE(3, 3, 3, 3));
		__m128 Vec3 = _mm_shuffle_ps(Temp3, Temp3, _MM_SHUFFLE(2, 2, 2, 0));

		// col0
		// + (Vec1[0] * Fac0[0] - Vec2[0] * Fac1[0] + Vec3[0] * Fac2[0]),
		// - (Vec1[1] * Fac0[1] - Vec2[1] * Fac1[1] + Vec3[1] * Fac2[1]),
		// + (Vec1[2] * Fac0[2] - Vec2[2] * Fac1[2] + Vec3[2] * Fac2[2]),
		// - (Vec1[3] * Fac0[3] - Vec2[3] * Fac1[3] + Vec3[3] * Fac2[3]),
		__m128 Mul00 = _mm_mul_ps(Vec1, Fac0);
		__m128 Mul01 = _mm_mul_ps(Vec2, Fac1);
		__m128 Mul02 = _mm_mul_ps(Vec3, Fac2);
		__m128 Sub00 = _mm_sub_ps(Mul00, Mul01);
		__m128 Add00 = _mm_add_ps(Sub00, Mul02);
		__m128 Inv0 = _mm_mul_ps(SignB, Add00);

		// col1
		// - (Vec0[0] * Fac0[0] - Vec2[0] * Fac3[0] + Vec3[0] * Fac4[0]),
		// + (Vec0[0] * Fac0[1] - Vec2[1] * Fac3[1] + Vec3[1] * Fac4[1]),
		// - (Vec0[0] * Fac0[2] - Vec2[2] * Fac3[2] + Vec3[2] * Fac4[2]),
		// + (Vec0[0] * Fac0[3] - Vec2[3] * Fac3[3] + Vec3[3] * Fac4[3]),
		__m128 Mul03 = _mm_mul_ps(Vec0, Fac0);
		__m128 Mul04 = _mm_mul_ps(Vec2, Fac3);
		__m128 Mul05 = _mm_mul_ps(Vec3, Fac4);
		__m128 Sub01 = _mm_sub_ps(Mul03, Mul04);
		__m128 Add01 = _mm_add_ps(Sub01, Mul05);
		__m128 Inv1 = _mm_mul_ps(SignA, Add01);

		// col2
		// + (Vec0[0] * Fac1[0] - Vec1[0] * Fac3[0] + Vec3[0] * Fac5[0]),
		// - (Vec0[0] * Fac1[1] - Vec1[1] * Fac3[1] + Vec3[1] * Fac5[1]),
		// + (Vec0[0] * Fac1[2] - Vec1[2] * Fac3[2] + Vec3[2] * Fac5[2]),
		// - (Vec0[0] * Fac1[3] - Vec1[3] * Fac3[3] + Vec3[3] * Fac5[3]),
		__m128 Mul06 = _mm_mul_ps(Vec0, Fac1);
		__m128 Mul07 = _mm_mul_ps(Vec1, Fac3);
		__m128 Mul08 = _mm_mul_ps(Vec3, Fac5);
		__m128 Sub02 = _mm_sub_ps(Mul06, Mul07);
		__m128 Add02 = _mm_add_ps(Sub02, Mul08);
		__m128 Inv2 = _mm_mul_ps(SignB, Add02);

		// col3
		// - (Vec1[0] * Fac2[0] - Vec1[0] * Fac4[0] + Vec2[0] * Fac5[0]),
		// + (Vec1[0] * Fac2[1] - Vec1[1] * Fac4[1] + Vec2[1] * Fac5[1]),
		// - (Vec1[0] * Fac2[2] - Vec1[2] * Fac4[2] + Vec2[2] * Fac5[2]),
		// + (Vec1[0] * Fac2[3] - Vec1[3] * Fac4[3] + Vec2[3] * Fac5[3]));
		__m128 Mul09 = _mm_mul_ps(Vec0, Fac2);
		__m128 Mul10 = _mm_mul_ps(Vec1, Fac4);
		__m128 Mul11 = _mm_mul_ps(Vec2, Fac5);
		__m128 Sub03 = _mm_sub_ps(Mul09, Mul10);
		__m128 Add03 = _mm_add_ps(Sub03, Mul11);
		__m128 Inv3 = _mm_mul_ps(SignA, Add03);

		__m128 Row0 = _mm_shuffle_ps(Inv0, Inv1, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Row1 = _mm_shuffle_ps(Inv2, Inv3, _MM_SHUFFLE(0, 0, 0, 0));
		__m128 Row2 = _mm_shuffle_ps(Row0, Row1, _MM_SHUFFLE(2, 0, 2, 0));

		//	valType Determinant = m[0][0] * Inverse[0][0]
		//						+ m[0][1] * Inverse[1][0]
		//						+ m[0][2] * Inverse[2][0]
		//						+ m[0][3] * Inverse[3][0];
		__m128 Det0 = _mm_dp_ps(cols[0], Row2, 0xff);
		__m128 Rcp0 = _mm_rcp_ps(Det0);
		//__m128 Rcp0 = _mm_div_ps(one, Det0);
		//	Inverse /= Determinant;

		matrix4 out;
		out[0] = _mm_mul_ps(Inv0, Rcp0);
		out[1] = _mm_mul_ps(Inv1, Rcp0);
		out[2] = _mm_mul_ps(Inv2, Rcp0);
		out[3] = _mm_mul_ps(Inv3, Rcp0);
		return out;
	}

	matrix4 transposed() const
	{
		__m128 tmp0 = _mm_shuffle_ps(cols[0], cols[1], 0x44);
		__m128 tmp2 = _mm_shuffle_ps(cols[0], cols[1], 0xEE);
		__m128 tmp1 = _mm_shuffle_ps(cols[2], cols[3], 0x44);
		__m128 tmp3 = _mm_shuffle_ps(cols[2], cols[3], 0xEE);

		matrix4 out;
		out[0] = _mm_shuffle_ps(tmp0, tmp1, 0x88);
		out[1] = _mm_shuffle_ps(tmp0, tmp1, 0xDD);
		out[2] = _mm_shuffle_ps(tmp2, tmp3, 0x88);
		out[3] = _mm_shuffle_ps(tmp2, tmp3, 0xDD);
		return out;
	}
};

struct vector4
{
	union {
		glm::vec4 vec;
		__m128 vec_4;
	};

	inline vector4(const __m128 &a4) : vec_4(a4) {}
	inline vector4(const __m128i &a4) : vec_4(_mm_castsi128_ps(a4)) {}
	inline explicit operator __m128() const { return vec_4; }

	inline static vector4 zero() { return vector4(_mm_setzero_ps()); }

	inline vector4() = default;
	inline vector4(const float *a) : vec_4(_mm_load_ps(a)) {}
	inline vector4(const float a) : vec_4(_mm_set1_ps(a)) {}
	inline vector4(const float a, const float b, const float c, const float d) : vec_4(_mm_set_ps(d, c, b, a)) {}
	inline vector4(const float *a, const simd::vector4 &mask) : vec_4(_mm_maskload_ps(a, _mm_castps_si128(mask.vec_4)))
	{
	}
	inline vector4(const glm::vec2 &v1, const glm::vec2 &v2) { vec_4 = _mm_set_ps(v2.y, v2.x, v1.y, v1.x); }
	inline vector4(const glm::vec3 &v, const float w) { vec_4 = _mm_set_ps(w, v.z, v.y, v.x); }
	inline vector4(const glm::vec3 &v) { vec_4 = _mm_set_ps(0, v.z, v.y, v.x); }
	inline vector4(const glm::vec4 &v) { vec_4 = _mm_set_ps(v.w, v.z, v.y, v.x); }

	inline explicit operator glm::vec4 &() { return vec; }
	inline explicit operator const glm::vec4 &() const { return vec; }

	inline const float &x() const { return vec.x; }
	inline const float &y() const { return vec.y; }
	inline const float &z() const { return vec.z; }
	inline const float &w() const { return vec.w; }

	inline float &x() { return vec.x; }
	inline float &y() { return vec.y; }
	inline float &z() { return vec.z; }
	inline float &w() { return vec.w; }

	inline float dot(const vector4 &op) const { return vector4(_mm_dp_ps(vec_4, op.vec_4, 0xff))[0]; }
	inline float length() const
	{
		union {
			__m128 result4;
			float result[4];
		};

		result4 = _mm_sqrt_ps(_mm_dp_ps(vec_4, vec_4, 0xff));
		return result[0];
	}
	inline vector4 min(const vector4 &op) const { return _mm_min_ps(vec_4, op.vec_4); }
	inline vector4 max(const vector4 &op) const { return _mm_max_ps(vec_4, op.vec_4); }
	inline vector4 abs() const { return _mm_castsi128_ps(_mm_abs_epi32(_mm_castps_si128(vec_4))); }
	inline vector4 sqrt() const { return _mm_sqrt_ps(vec_4); }
	inline vector4 inv_sqrt() const { return _mm_rsqrt_ps(vec_4); }

	inline vector4 operator*(const vector4 &op) const { return _mm_mul_ps(vec_4, op.vec_4); }
	inline vector4 operator/(const vector4 &op) const { return _mm_div_ps(vec_4, op.vec_4); }
	inline vector4 operator+(const vector4 &op) const { return _mm_add_ps(vec_4, op.vec_4); }
	inline vector4 operator-(const vector4 &op) const { return _mm_sub_ps(vec_4, op.vec_4); }
	inline vector4 operator>(const vector4 &op) const { return _mm_cmpgt_ps(vec_4, op.vec_4); }
	inline vector4 operator<(const vector4 &op) const { return _mm_cmplt_ps(vec_4, op.vec_4); }
	inline vector4 operator>=(const vector4 &op) const { return _mm_cmpge_ps(vec_4, op.vec_4); }
	inline vector4 operator<=(const vector4 &op) const { return _mm_cmple_ps(vec_4, op.vec_4); }
	inline vector4 operator==(const vector4 &op) const { return _mm_cmpeq_ps(vec_4, op.vec_4); }
	inline vector4 operator!=(const vector4 &op) const { return _mm_cmpneq_ps(vec_4, op.vec_4); }
	inline vector4 operator&(const vector4 &op) const { return _mm_and_ps(vec_4, op.vec_4); }
	inline vector4 operator|(const vector4 &op) const { return _mm_or_ps(vec_4, op.vec_4); }

	inline vector4 operator*(const __m128 &op) const { return _mm_mul_ps(vec_4, op); }
	inline vector4 operator/(const __m128 &op) const { return _mm_div_ps(vec_4, op); }
	inline vector4 operator+(const __m128 &op) const { return _mm_add_ps(vec_4, op); }
	inline vector4 operator-(const __m128 &op) const { return _mm_sub_ps(vec_4, op); }
	inline vector4 operator>(const __m128 &op) const { return _mm_cmpgt_ps(vec_4, op); }
	inline vector4 operator<(const __m128 &op) const { return _mm_cmplt_ps(vec_4, op); }
	inline vector4 operator>=(const __m128 &op) const { return _mm_cmpge_ps(vec_4, op); }
	inline vector4 operator<=(const __m128 &op) const { return _mm_cmple_ps(vec_4, op); }
	inline vector4 operator==(const __m128 &op) const { return _mm_cmpeq_ps(vec_4, op); }
	inline vector4 operator!=(const __m128 &op) const { return _mm_cmpneq_ps(vec_4, op); }
	inline vector4 operator&(const __m128 &op) const { return _mm_and_ps(vec_4, op); }
	inline vector4 operator|(const __m128 &op) const { return _mm_or_ps(vec_4, op); }

	inline void operator*=(const vector4 &op) { vec_4 = _mm_mul_ps(vec_4, op.vec_4); }
	inline void operator/=(const vector4 &op) { vec_4 = _mm_div_ps(vec_4, op.vec_4); }
	inline void operator+=(const vector4 &op) { vec_4 = _mm_add_ps(vec_4, op.vec_4); }
	inline void operator-=(const vector4 &op) { vec_4 = _mm_sub_ps(vec_4, op.vec_4); }
	inline void operator&=(const vector4 &op) { vec_4 = _mm_and_ps(vec_4, op.vec_4); }
	inline void operator|=(const vector4 &op) { vec_4 = _mm_or_ps(vec_4, op.vec_4); }

	inline void operator*=(const __m128 &op) { vec_4 = _mm_mul_ps(vec_4, op); }
	inline void operator/=(const __m128 &op) { vec_4 = _mm_div_ps(vec_4, op); }
	inline void operator+=(const __m128 &op) { vec_4 = _mm_add_ps(vec_4, op); }
	inline void operator-=(const __m128 &op) { vec_4 = _mm_sub_ps(vec_4, op); }
	inline void operator&=(const __m128 &op) { vec_4 = _mm_and_ps(vec_4, op); }
	inline void operator|=(const __m128 &op) { vec_4 = _mm_or_ps(vec_4, op); }

	inline const float &operator[](int index) const { return vec[index]; }
	inline float &operator[](int index) { return vec[index]; }

	inline int move_mask() const { return _mm_movemask_ps(vec_4); }
	inline glm::bvec4 to_mask() const
	{
		const int mask = move_mask();
		return glm::bvec4(mask & 1, mask & 2, mask & 4, mask & 8);
	}

	inline void write_to(float *loc) const { _mm_store_ps(loc, vec_4); }
	inline void write_to(float *loc, const vector4 &mask) const
	{
		_mm_maskstore_ps(loc, _mm_castps_si128(mask.vec_4), vec_4);
	}
	inline void store(const vector4 &result, const vector4 &mask)
	{
		_mm_maskstore_ps(value_ptr(vec), _mm_castps_si128(mask.vec_4), result.vec_4);
	}
};

inline vector4 min(const vector4 &l, const vector4 &r) { return _mm_min_ps(l.vec_4, r.vec_4); }
inline vector4 max(const vector4 &l, const vector4 &r) { return _mm_max_ps(l.vec_4, r.vec_4); }
inline float dot(const vector4 &l, const vector4 &r) { return vector4(_mm_dp_ps(l.vec_4, r.vec_4, 0xff))[0]; }
inline vector4 abs(const vector4 &op) { return _mm_castsi128_ps(_mm_abs_epi32(_mm_castps_si128(op.vec_4))); }

inline glm::vec4 operator*(const glm::vec4 &op, const matrix4 &mat)
{
#if 1
	__m256 i01 = mat.cols8[0];
	__m256 i23 = mat.cols8[1];

	const __m128 v = _mm_set_ps(op.w, op.z, op.y, op.x);
	const __m256 v8 = _mm256_set_m128(v, v);

	__m256 m01 = _mm256_mul_ps(v8, i01);
	__m256 m23 = _mm256_mul_ps(v8, i23);

	__m128 u0 = _mm_unpacklo_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 u1 = _mm_unpackhi_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 u3 = _mm_unpackhi_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);
	union {
		glm::vec4 f2v;
		__m128 f2;
	};

	f2 = _mm_add_ps(f0, f1);
	return f2v;
#else
	__m128 i0 = mat.cols[0];
	__m128 i1 = mat.cols[1];
	__m128 i2 = mat.cols[2];
	__m128 i3 = mat.cols[3];

	const __m128 v = _mm_set_ps(op.w, op.z, op.y, op.x);

	__m128 m0 = _mm_mul_ps(v, i0);
	__m128 m1 = _mm_mul_ps(v, i1);
	__m128 m2 = _mm_mul_ps(v, i2);
	__m128 m3 = _mm_mul_ps(v, i3);

	__m128 u0 = _mm_unpacklo_ps(m0, m1);
	__m128 u1 = _mm_unpackhi_ps(m0, m1);
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(m2, m3);
	__m128 u3 = _mm_unpackhi_ps(m2, m3);
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);
	union {
		glm::vec4 f2v;
		__m128 f2;
	};

	f2 = _mm_add_ps(f0, f1);
	return f2v;
#endif
}

inline vector4 operator*(const vector4 &op, const matrix4 &mat)
{
#if 1
	__m256 i01 = mat.cols8[0];
	__m256 i23 = mat.cols8[1];

	const __m128 &v = op.vec_4;
	const __m256 v8 = _mm256_set_m128(v, v);

	__m256 m01 = _mm256_mul_ps(v8, i01);
	__m256 m23 = _mm256_mul_ps(v8, i23);

	__m128 u0 = _mm_unpacklo_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 u1 = _mm_unpackhi_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 u3 = _mm_unpackhi_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);

	return _mm_add_ps(f0, f1);
#else
	__m128 i0 = mat.cols[0];
	__m128 i1 = mat.cols[1];
	__m128 i2 = mat.cols[2];
	__m128 i3 = mat.cols[3];

	const __m128 &v = op.vec_4;

	__m128 m0 = _mm_mul_ps(v, i0);
	__m128 m1 = _mm_mul_ps(v, i1);
	__m128 m2 = _mm_mul_ps(v, i2);
	__m128 m3 = _mm_mul_ps(v, i3);

	__m128 u0 = _mm_unpacklo_ps(m0, m1);
	__m128 u1 = _mm_unpackhi_ps(m0, m1);
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(m2, m3);
	__m128 u3 = _mm_unpackhi_ps(m2, m3);
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);
	union {
		glm::vec4 f2v;
		__m128 f2;
	};

	f2 = _mm_add_ps(f0, f1);
	return f2v;
#endif
}

inline glm::vec4 operator*(const matrix4 &mat, const glm::vec4 &op)
{
	const __m128 v = _mm_set_ps(op.w, op.z, op.y, op.x);

	__m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 v1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 v2 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 v3 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 m0 = _mm_mul_ps(mat.cols[0], v0);
	__m128 m1 = _mm_mul_ps(mat.cols[1], v1);
	__m128 m2 = _mm_mul_ps(mat.cols[2], v2);
	__m128 m3 = _mm_mul_ps(mat.cols[3], v3);

	__m128 a0 = _mm_add_ps(m0, m1);
	__m128 a1 = _mm_add_ps(m2, m3);
	union {
		glm::vec4 a2v;
		__m128 a2;
	};
	a2 = _mm_add_ps(a0, a1);
	return a2v;
}

inline vector4 operator*(const matrix4 &mat, const vector4 &op)
{
	const __m128 &v = op.vec_4;

	__m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 v1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 v2 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 v3 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 m0 = _mm_mul_ps(mat.cols[0], v0);
	__m128 m1 = _mm_mul_ps(mat.cols[1], v1);
	__m128 m2 = _mm_mul_ps(mat.cols[2], v2);
	__m128 m3 = _mm_mul_ps(mat.cols[3], v3);

	__m128 a0 = _mm_add_ps(m0, m1);
	__m128 a1 = _mm_add_ps(m2, m3);
	union {
		glm::vec4 a2v;
		__m128 a2;
	};
	a2 = _mm_add_ps(a0, a1);
	return a2v;
}

inline vector4 operator*(const vector4 &op1, const float op2) { return _mm_mul_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator/(const vector4 &op1, const float op2) { return _mm_div_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator+(const vector4 &op1, const float op2) { return _mm_add_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator-(const vector4 &op1, const float op2) { return _mm_sub_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator>(const vector4 &op1, const float op2) { return _mm_cmpgt_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator<(const vector4 &op1, const float op2) { return _mm_cmplt_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator>=(const vector4 &op1, const float op2) { return _mm_cmpge_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator<=(const vector4 &op1, const float op2) { return _mm_cmple_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator==(const vector4 &op1, const float op2) { return _mm_cmpeq_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator!=(const vector4 &op1, const float op2) { return _mm_cmpneq_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator&(const vector4 &op1, const float op2) { return _mm_and_ps(op1.vec_4, _mm_set1_ps(op2)); }
inline vector4 operator|(const vector4 &op1, const float op2) { return _mm_or_ps(op1.vec_4, _mm_set1_ps(op2)); }

inline vector4 operator*(const float op1, const vector4 &op2) { return _mm_mul_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator/(const float op1, const vector4 &op2) { return _mm_div_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator+(const float op1, const vector4 &op2) { return _mm_add_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator-(const float op1, const vector4 &op2) { return _mm_sub_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator>(const float op1, const vector4 &op2) { return _mm_cmpgt_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator<(const float op1, const vector4 &op2) { return _mm_cmplt_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator>=(const float op1, const vector4 &op2) { return _mm_cmpge_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator<=(const float op1, const vector4 &op2) { return _mm_cmple_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator==(const float op1, const vector4 &op2) { return _mm_cmpeq_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator!=(const float op1, const vector4 &op2) { return _mm_cmpneq_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator&(const float op1, const vector4 &op2) { return _mm_and_ps(_mm_set1_ps(op1), op2.vec_4); }
inline vector4 operator|(const float op1, const vector4 &op2) { return _mm_or_ps(_mm_set1_ps(op1), op2.vec_4); }

inline vector4 operator*(const vector4 &op1, const __m128 &op2) { return _mm_mul_ps(op1.vec_4, op2); }
inline vector4 operator/(const vector4 &op1, const __m128 &op2) { return _mm_div_ps(op1.vec_4, op2); }
inline vector4 operator+(const vector4 &op1, const __m128 &op2) { return _mm_add_ps(op1.vec_4, op2); }
inline vector4 operator-(const vector4 &op1, const __m128 &op2) { return _mm_sub_ps(op1.vec_4, op2); }
inline vector4 operator>(const vector4 &op1, const __m128 &op2) { return _mm_cmpgt_ps(op1.vec_4, op2); }
inline vector4 operator<(const vector4 &op1, const __m128 &op2) { return _mm_cmplt_ps(op1.vec_4, op2); }
inline vector4 operator>=(const vector4 &op1, const __m128 &op2) { return _mm_cmpge_ps(op1.vec_4, op2); }
inline vector4 operator<=(const vector4 &op1, const __m128 &op2) { return _mm_cmple_ps(op1.vec_4, op2); }
inline vector4 operator==(const vector4 &op1, const __m128 &op2) { return _mm_cmpeq_ps(op1.vec_4, op2); }
inline vector4 operator!=(const vector4 &op1, const __m128 &op2) { return _mm_cmpneq_ps(op1.vec_4, op2); }
inline vector4 operator&(const vector4 &op1, const __m128 &op2) { return _mm_and_ps(op1.vec_4, op2); }
inline vector4 operator|(const vector4 &op1, const __m128 &op2) { return _mm_or_ps(op1.vec_4, op2); }

inline vector4 operator*(const __m128 &op1, const vector4 &op2) { return _mm_mul_ps(op1, op2.vec_4); }
inline vector4 operator/(const __m128 &op1, const vector4 &op2) { return _mm_div_ps(op1, op2.vec_4); }
inline vector4 operator+(const __m128 &op1, const vector4 &op2) { return _mm_add_ps(op1, op2.vec_4); }
inline vector4 operator-(const __m128 &op1, const vector4 &op2) { return _mm_sub_ps(op1, op2.vec_4); }
inline vector4 operator>(const __m128 &op1, const vector4 &op2) { return _mm_cmpgt_ps(op1, op2.vec_4); }
inline vector4 operator<(const __m128 &op1, const vector4 &op2) { return _mm_cmplt_ps(op1, op2.vec_4); }
inline vector4 operator>=(const __m128 &op1, const vector4 &op2) { return _mm_cmpge_ps(op1, op2.vec_4); }
inline vector4 operator<=(const __m128 &op1, const vector4 &op2) { return _mm_cmple_ps(op1, op2.vec_4); }
inline vector4 operator==(const __m128 &op1, const vector4 &op2) { return _mm_cmpeq_ps(op1, op2.vec_4); }
inline vector4 operator!=(const __m128 &op1, const vector4 &op2) { return _mm_cmpneq_ps(op1, op2.vec_4); }
inline vector4 operator&(const __m128 &op1, const vector4 &op2) { return _mm_and_ps(op1, op2.vec_4); }
inline vector4 operator|(const __m128 &op1, const vector4 &op2) { return _mm_or_ps(op1, op2.vec_4); }

struct vector8
{
	union {
		glm::vec4 vec[2];
		__m128 vec_4[2];
		vector4 vec4[2];
		__m256 vec_8;
	};

	inline vector8(const vector4 &a, const vector4 &b)
	{
		vec4[0] = a;
		vec4[1] = b;
	}
	inline vector8(const __m256 &a) : vec_8(a) {}
	inline vector8(const __m128 &a4, const __m128 &b4) : vec_8(_mm256_set_m128(b4, a4)) {}
	inline vector8(const __m128i &a4, const __m128i &b4) : vector8(_mm_castsi128_ps(a4), _mm_castsi128_ps(b4)) {}
	inline explicit operator __m256() const { return vec_8; }

	inline static vector8 zero() { return vector8(_mm256_setzero_ps()); }

	inline vector8() = default;
	inline vector8(const float *a) : vec_8(_mm256_load_ps(a)) {}
	inline vector8(const float a) : vec_8(_mm256_set1_ps(a)) {}
	inline vector8(const float a, const float b, const float c, const float d, const float e, const float f,
				   const float g, const float h)
		: vec_8(_mm256_set_ps(h, g, f, e, d, c, b, a))
	{
	}
	inline vector8(const glm::vec2 &v1, const glm::vec2 &v2, const glm::vec2 &v3, const glm::vec2 &v4)
	{
		vec_8 = _mm256_set_ps(v4.y, v4.x, v3.y, v3.x, v2.y, v2.x, v1.y, v1.x);
	}
	inline vector8(const glm::vec3 &v, const float w, const glm::vec3 &v2, const float w2)
	{
		vec_8 = _mm256_set_ps(w2, v2.z, v2.y, v2.x, w, v.z, v.y, v.x);
	}
	inline vector8(const glm::vec4 &v, const glm::vec4 &v2)
	{
		vec_8 = _mm256_set_ps(v2.w, v2.z, v2.y, v2.x, v.w, v.z, v.y, v.x);
	}

	inline const float &x1() const { return vec[0].x; }
	inline const float &y1() const { return vec[0].y; }
	inline const float &z1() const { return vec[0].z; }
	inline const float &w1() const { return vec[0].w; }

	inline const float &x2() const { return vec[1].x; }
	inline const float &y2() const { return vec[1].y; }
	inline const float &z2() const { return vec[1].z; }
	inline const float &w2() const { return vec[1].w; }

	inline const vector4 &get_vector4(int idx) const { return vec4[idx]; }

	inline float &x1() { return vec[0].x; }
	inline float &y1() { return vec[0].y; }
	inline float &z1() { return vec[0].z; }
	inline float &w1() { return vec[0].w; }

	inline float &x2() { return vec[1].x; }
	inline float &y2() { return vec[1].y; }
	inline float &z2() { return vec[1].z; }
	inline float &w2() { return vec[1].w; }

	inline vector8 min(const vector8 &op) const { return _mm256_min_ps(vec_8, op.vec_8); }
	inline vector8 max(const vector8 &op) const { return _mm256_max_ps(vec_8, op.vec_8); }

	inline vector8 operator*(const vector8 &op) const { return _mm256_mul_ps(vec_8, op.vec_8); }
	inline vector8 operator/(const vector8 &op) const { return _mm256_div_ps(vec_8, op.vec_8); }
	inline vector8 operator+(const vector8 &op) const { return _mm256_add_ps(vec_8, op.vec_8); }
	inline vector8 operator-(const vector8 &op) const { return _mm256_sub_ps(vec_8, op.vec_8); }
	inline vector8 operator>(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpgt_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmpgt_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator<(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmplt_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmplt_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator>=(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpge_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmpge_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator<=(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmple_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmple_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator==(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpeq_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmpeq_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator!=(const vector8 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpneq_ps(vec_4[0], op.vec_4[0]);
		result.vec_4[1] = _mm_cmpneq_ps(vec_4[1], op.vec_4[1]);
		return result;
	}
	inline vector8 operator&(const vector8 &op) const { return _mm256_and_ps(vec_8, op.vec_8); }
	inline vector8 operator|(const vector8 &op) const { return _mm256_or_ps(vec_8, op.vec_8); }

	inline vector8 operator*(const __m256 &op) const { return _mm256_mul_ps(vec_8, op); }
	inline vector8 operator/(const __m256 &op) const { return _mm256_div_ps(vec_8, op); }
	inline vector8 operator+(const __m256 &op) const { return _mm256_add_ps(vec_8, op); }
	inline vector8 operator-(const __m256 &op) const { return _mm256_sub_ps(vec_8, op); }
	inline vector8 operator>(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpgt_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmpgt_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator<(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmplt_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmplt_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator>=(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpge_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmpge_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator<=(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmple_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmple_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator==(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpeq_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmpeq_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator!=(const __m256 &op) const
	{
		vector8 result;
		result.vec_4[0] = _mm_cmpneq_ps(vec_4[0], _mm256_extractf128_ps(op, 0));
		result.vec_4[1] = _mm_cmpneq_ps(vec_4[1], _mm256_extractf128_ps(op, 1));
		return result;
	}
	inline vector8 operator&(const __m256 &op) const { return _mm256_and_ps(vec_8, op); }
	inline vector8 operator|(const __m256 &op) const { return _mm256_or_ps(vec_8, op); }

	inline void operator*=(const vector8 &op) { vec_8 = _mm256_mul_ps(vec_8, op.vec_8); }
	inline void operator/=(const vector8 &op) { vec_8 = _mm256_div_ps(vec_8, op.vec_8); }
	inline void operator+=(const vector8 &op) { vec_8 = _mm256_add_ps(vec_8, op.vec_8); }
	inline void operator-=(const vector8 &op) { vec_8 = _mm256_sub_ps(vec_8, op.vec_8); }
	inline void operator&=(const vector8 &op) { vec_8 = _mm256_and_ps(vec_8, op.vec_8); }
	inline void operator|=(const vector8 &op) { vec_8 = _mm256_or_ps(vec_8, op.vec_8); }

	inline void operator*=(const __m256 &op) { vec_8 = _mm256_mul_ps(vec_8, op); }
	inline void operator/=(const __m256 &op) { vec_8 = _mm256_div_ps(vec_8, op); }
	inline void operator+=(const __m256 &op) { vec_8 = _mm256_add_ps(vec_8, op); }
	inline void operator-=(const __m256 &op) { vec_8 = _mm256_sub_ps(vec_8, op); }
	inline void operator&=(const __m256 &op) { vec_8 = _mm256_and_ps(vec_8, op); }
	inline void operator|=(const __m256 &op) { vec_8 = _mm256_or_ps(vec_8, op); }

	inline int move_mask() const { return _mm256_movemask_ps(vec_8); }

	inline void write_to(float *loc) const { _mm256_store_ps(loc, vec_8); }
	inline void write_to(float *loc, const vector8 &mask) const
	{
		_mm256_maskstore_ps(loc, _mm256_castps_si256(mask.vec_8), vec_8);
	}
	inline void store(const vector8 &result, const vector8 &mask)
	{
		_mm_maskstore_ps(value_ptr(vec[0]), _mm_castps_si128(mask.vec_4[0]), result.vec_4[0]);
		_mm_maskstore_ps(value_ptr(vec[1]), _mm_castps_si128(mask.vec_4[1]), result.vec_4[1]);
	}
};

inline vector8 operator*(const vector8 &op1, const float op2) { return _mm256_mul_ps(op1.vec_8, _mm256_set1_ps(op2)); }
inline vector8 operator/(const vector8 &op1, const float op2) { return _mm256_div_ps(op1.vec_8, _mm256_set1_ps(op2)); }
inline vector8 operator+(const vector8 &op1, const float op2) { return _mm256_add_ps(op1.vec_8, _mm256_set1_ps(op2)); }
inline vector8 operator-(const vector8 &op1, const float op2) { return _mm256_sub_ps(op1.vec_8, _mm256_set1_ps(op2)); }
inline vector8 operator>(const vector8 &op1, const float op2)
{

	vector8 result;
	result.vec_4[0] = _mm_cmpgt_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmpgt_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator<(const vector8 &op1, const float op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmplt_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmplt_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator>=(const vector8 &op1, const float op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpge_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmpge_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator<=(const vector8 &op1, const float op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmplt_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmplt_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator==(const vector8 &op1, const float op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpeq_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmpeq_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator!=(const vector8 &op1, const float op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpneq_ps(op1.vec_4[0], _mm_set1_ps(op2));
	result.vec_4[1] = _mm_cmpneq_ps(op1.vec_4[1], _mm_set1_ps(op2));
	return result;
}
inline vector8 operator&(const vector8 &op1, const float op2) { return _mm256_and_ps(op1.vec_8, _mm256_set1_ps(op2)); }
inline vector8 operator|(const vector8 &op1, const float op2) { return _mm256_or_ps(op1.vec_8, _mm256_set1_ps(op2)); }

inline vector8 operator*(const float op1, const vector8 &op2) { return _mm256_mul_ps(_mm256_set1_ps(op1), op2.vec_8); }
inline vector8 operator/(const float op1, const vector8 &op2) { return _mm256_div_ps(_mm256_set1_ps(op1), op2.vec_8); }
inline vector8 operator+(const float op1, const vector8 &op2) { return _mm256_add_ps(_mm256_set1_ps(op1), op2.vec_8); }
inline vector8 operator-(const float op1, const vector8 &op2) { return _mm256_sub_ps(_mm256_set1_ps(op1), op2.vec_8); }
inline vector8 operator>(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpgt_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpgt_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator<(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmplt_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmplt_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator>=(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpge_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpge_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator<=(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmple_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmple_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator==(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpeq_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpeq_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator!=(const float op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpneq_ps(_mm_set1_ps(op1), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpneq_ps(_mm_set1_ps(op1), op2.vec_4[1]);
	return result;
}
inline vector8 operator&(const float op1, const vector8 &op2) { return _mm256_and_ps(_mm256_set1_ps(op1), op2.vec_8); }
inline vector8 operator|(const float op1, const vector8 &op2) { return _mm256_or_ps(_mm256_set1_ps(op1), op2.vec_8); }

inline vector8 operator*(const vector8 &op1, const __m256 &op2) { return _mm256_mul_ps(op1.vec_8, op2); }
inline vector8 operator/(const vector8 &op1, const __m256 &op2) { return _mm256_div_ps(op1.vec_8, op2); }
inline vector8 operator+(const vector8 &op1, const __m256 &op2) { return _mm256_add_ps(op1.vec_8, op2); }
inline vector8 operator-(const vector8 &op1, const __m256 &op2) { return _mm256_sub_ps(op1.vec_8, op2); }
inline vector8 operator>(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpgt_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmpgt_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator<(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmplt_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmplt_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator>=(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpge_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmpge_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator<=(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmple_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmple_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator==(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpeq_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmpeq_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator!=(const vector8 &op1, const __m256 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpneq_ps(_mm256_extractf128_ps(op2, 0), op1.vec_4[0]);
	result.vec_4[1] = _mm_cmpneq_ps(_mm256_extractf128_ps(op2, 1), op1.vec_4[1]);
	return result;
}
inline vector8 operator&(const vector8 &op1, const __m256 &op2) { return _mm256_and_ps(op1.vec_8, op2); }
inline vector8 operator|(const vector8 &op1, const __m256 &op2) { return _mm256_or_ps(op1.vec_8, op2); }

inline vector8 operator*(const __m256 &op1, const vector8 &op2) { return _mm256_mul_ps(op1, op2.vec_8); }
inline vector8 operator/(const __m256 &op1, const vector8 &op2) { return _mm256_div_ps(op1, op2.vec_8); }
inline vector8 operator+(const __m256 &op1, const vector8 &op2) { return _mm256_add_ps(op1, op2.vec_8); }
inline vector8 operator-(const __m256 &op1, const vector8 &op2) { return _mm256_sub_ps(op1, op2.vec_8); }
inline vector8 operator>(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpgt_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpgt_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator<(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmplt_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmplt_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator>=(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpge_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpge_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator<=(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmple_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmple_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator==(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpeq_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpeq_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator!=(const __m256 &op1, const vector8 &op2)
{
	vector8 result;
	result.vec_4[0] = _mm_cmpneq_ps(_mm256_extractf128_ps(op1, 0), op2.vec_4[0]);
	result.vec_4[1] = _mm_cmpneq_ps(_mm256_extractf128_ps(op1, 1), op2.vec_4[1]);
	return result;
}
inline vector8 operator&(const __m256 &op1, const vector8 &op2) { return _mm256_and_ps(op1, op2.vec_8); }
inline vector8 operator|(const __m256 &op1, const vector8 &op2) { return _mm256_or_ps(op1, op2.vec_8); }

inline vector8 min(const vector8 &a, const vector8 &b) { return _mm256_min_ps(a.vec_8, b.vec_8); }
inline vector8 max(const vector8 &a, const vector8 &b) { return _mm256_max_ps(a.vec_8, b.vec_8); }

inline vector4 log(const vector4 &op)
{
	__m128 x = op.vec_4;
	__m128i emm0;

	__m128 one = *(__m128 *)_ps_1;

	__m128 invalid_mask = _mm_cmple_ps(x, _mm_setzero_ps());

	x = _mm_max_ps(x, *(__m128 *)_ps_min_norm_pos); /* cut off denormalized stuff */

	emm0 = _mm_srli_epi32(_mm_castps_si128(x), 23);

	/* keep only the fractional part */
	x = _mm_and_ps(x, *(__m128 *)_ps_inv_mant_mask);
	x = _mm_or_ps(x, *(__m128 *)_ps_0p5);

	emm0 = _mm_sub_epi32(emm0, *(__m128i *)_pi32_0x7f);
	__m128 e = _mm_cvtepi32_ps(emm0);

	e = _mm_add_ps(e, one);

	/* part2:
	   if( x < SQRTHF ) {
		 e -= 1;
		 x = x + x - 1.0;
	   } else { x = x - 1.0; }
	*/
	__m128 mask = _mm_cmplt_ps(x, *(__m128 *)_ps_cephes_SQRTHF);
	__m128 tmp = _mm_and_ps(x, mask);
	x = _mm_sub_ps(x, one);
	e = _mm_sub_ps(e, _mm_and_ps(one, mask));
	x = _mm_add_ps(x, tmp);

	__m128 z = _mm_mul_ps(x, x);

	__m128 y = *(__m128 *)_ps_cephes_log_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p5);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p6);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p7);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_log_p8);
	y = _mm_mul_ps(y, x);

	y = _mm_mul_ps(y, z);

	tmp = _mm_mul_ps(e, *(__m128 *)_ps_cephes_log_q1);
	y = _mm_add_ps(y, tmp);

	tmp = _mm_mul_ps(z, *(__m128 *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);

	tmp = _mm_mul_ps(e, *(__m128 *)_ps_cephes_log_q2);
	x = _mm_add_ps(x, y);
	x = _mm_add_ps(x, tmp);
	x = _mm_or_ps(x, invalid_mask); // negative arg will be NAN
	return x;
}
inline vector4 exp(const vector4 &op)
{
	__m128 x = op.vec_4;
	__m128 tmp = _mm_setzero_ps(), fx;
	__m128i emm0;
	__m128 one = *(__m128 *)_ps_1;

	x = _mm_min_ps(x, *(__m128 *)_ps_exp_hi);
	x = _mm_max_ps(x, *(__m128 *)_ps_exp_lo);

	/* express exp(x) as exp(g + n*log(2)) */
	fx = _mm_mul_ps(x, *(__m128 *)_ps_cephes_LOG2EF);
	fx = _mm_add_ps(fx, *(__m128 *)_ps_0p5);

	/* how to perform a floorf with SSE: just below */
	emm0 = _mm_cvttps_epi32(fx);
	tmp = _mm_cvtepi32_ps(emm0);
	/* if greater, substract 1 */
	__m128 mask = _mm_cmpgt_ps(tmp, fx);
	mask = _mm_and_ps(mask, one);
	fx = _mm_sub_ps(tmp, mask);

	tmp = _mm_mul_ps(fx, *(__m128 *)_ps_cephes_exp_C1);
	__m128 z = _mm_mul_ps(fx, *(__m128 *)_ps_cephes_exp_C2);
	x = _mm_sub_ps(x, tmp);
	x = _mm_sub_ps(x, z);

	z = _mm_mul_ps(x, x);

	__m128 y = *(__m128 *)_ps_cephes_exp_p0;
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_exp_p1);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_exp_p2);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_exp_p3);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_exp_p4);
	y = _mm_mul_ps(y, x);
	y = _mm_add_ps(y, *(__m128 *)_ps_cephes_exp_p5);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, x);
	y = _mm_add_ps(y, one);

	/* build 2^n */
	emm0 = _mm_cvttps_epi32(fx);
	emm0 = _mm_add_epi32(emm0, *(__m128i *)_pi32_0x7f);
	emm0 = _mm_slli_epi32(emm0, 23);
	__m128 pow2n = _mm_castsi128_ps(emm0);
	y = _mm_mul_ps(y, pow2n);
	return y;
}
inline vector4 sin(const vector4 &op)
{
	__m128 x = op.vec_4;
	// any x
	__m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, sign_bit, y;

	__m128i emm0, emm2;
	sign_bit = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(__m128 *)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit = _mm_and_ps(sign_bit, *(__m128 *)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(__m128 *)_ps_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(__m128i *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	/* get the swap sign flag */
	emm0 = _mm_and_si128(emm2, *(__m128i *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask
	   there is one polynom for 0 <= x <= Pi/4
	   and another one for Pi/4<x<=Pi/2

	   Both branches will be computed.
	*/
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	__m128 swap_sign_bit = _mm_castsi128_ps(emm0);
	__m128 poly_mask = _mm_castsi128_ps(emm2);
	sign_bit = _mm_xor_ps(sign_bit, swap_sign_bit);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(__m128 *)_ps_minus_cephes_DP1;
	xmm2 = *(__m128 *)_ps_minus_cephes_DP2;
	xmm3 = *(__m128 *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(__m128 *)_ps_coscof_p0;
	__m128 z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	__m128 tmp = _mm_mul_ps(z, *(__m128 *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(__m128 *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	__m128 y2 = *(__m128 *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);
	return y;
}
inline vector4 cos(const vector4 &op)
{
	__m128 x = op.vec_4;

	// any x
	__m128 xmm1, xmm2 = _mm_setzero_ps(), xmm3, y;
	__m128i emm0, emm2;
	/* take the absolute value */
	x = _mm_and_ps(x, *(__m128 *)_ps_inv_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(__m128 *)_ps_cephes_FOPI);

	/* store the integer part of y in mm0 */
	emm2 = _mm_cvttps_epi32(y);
	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(__m128i *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm2 = _mm_sub_epi32(emm2, *(__m128i *)_pi32_2);

	/* get the swap sign flag */
	emm0 = _mm_andnot_si128(emm2, *(__m128i *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	/* get the polynom selection mask */
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());

	__m128 sign_bit = _mm_castsi128_ps(emm0);
	__m128 poly_mask = _mm_castsi128_ps(emm2);
	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(__m128 *)_ps_minus_cephes_DP1;
	xmm2 = *(__m128 *)_ps_minus_cephes_DP2;
	xmm3 = *(__m128 *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	y = *(__m128 *)_ps_coscof_p0;
	__m128 z = _mm_mul_ps(x, x);

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	__m128 tmp = _mm_mul_ps(z, *(__m128 *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(__m128 *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	__m128 y2 = *(__m128 *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	y2 = _mm_and_ps(xmm3, y2); //, xmm3);
	y = _mm_andnot_ps(xmm3, y);
	y = _mm_add_ps(y, y2);
	/* update the sign */
	y = _mm_xor_ps(y, sign_bit);

	return y;
}
inline void sincos(const vector4 &op, vector4 *s, vector4 *c)
{
	__m128 x = op.vec_4;

	__m128 xmm1, xmm2, xmm3 = _mm_setzero_ps(), sign_bit_sin, y;
	__m128i emm0, emm2, emm4;
	sign_bit_sin = x;
	/* take the absolute value */
	x = _mm_and_ps(x, *(__m128 *)_ps_inv_sign_mask);
	/* extract the sign bit (upper one) */
	sign_bit_sin = _mm_and_ps(sign_bit_sin, *(__m128 *)_ps_sign_mask);

	/* scale by 4/Pi */
	y = _mm_mul_ps(x, *(__m128 *)_ps_cephes_FOPI);

	/* store the integer part of y in emm2 */
	emm2 = _mm_cvttps_epi32(y);

	/* j=(j+1) & (~1) (see the cephes sources) */
	emm2 = _mm_add_epi32(emm2, *(__m128i *)_pi32_1);
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_inv1);
	y = _mm_cvtepi32_ps(emm2);

	emm4 = emm2;

	/* get the swap sign flag for the sine */
	emm0 = _mm_and_si128(emm2, *(__m128i *)_pi32_4);
	emm0 = _mm_slli_epi32(emm0, 29);
	__m128 swap_sign_bit_sin = _mm_castsi128_ps(emm0);

	/* get the polynom selection mask for the sine*/
	emm2 = _mm_and_si128(emm2, *(__m128i *)_pi32_2);
	emm2 = _mm_cmpeq_epi32(emm2, _mm_setzero_si128());
	__m128 poly_mask = _mm_castsi128_ps(emm2);

	/* The magic pass: "Extended precision modular arithmetic"
	   x = ((x - y * DP1) - y * DP2) - y * DP3; */
	xmm1 = *(__m128 *)_ps_minus_cephes_DP1;
	xmm2 = *(__m128 *)_ps_minus_cephes_DP2;
	xmm3 = *(__m128 *)_ps_minus_cephes_DP3;
	xmm1 = _mm_mul_ps(y, xmm1);
	xmm2 = _mm_mul_ps(y, xmm2);
	xmm3 = _mm_mul_ps(y, xmm3);
	x = _mm_add_ps(x, xmm1);
	x = _mm_add_ps(x, xmm2);
	x = _mm_add_ps(x, xmm3);

	emm4 = _mm_sub_epi32(emm4, *(__m128i *)_pi32_2);
	emm4 = _mm_andnot_si128(emm4, *(__m128i *)_pi32_4);
	emm4 = _mm_slli_epi32(emm4, 29);
	__m128 sign_bit_cos = _mm_castsi128_ps(emm4);

	sign_bit_sin = _mm_xor_ps(sign_bit_sin, swap_sign_bit_sin);

	/* Evaluate the first polynom  (0 <= x <= Pi/4) */
	__m128 z = _mm_mul_ps(x, x);
	y = *(__m128 *)_ps_coscof_p0;

	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p1);
	y = _mm_mul_ps(y, z);
	y = _mm_add_ps(y, *(__m128 *)_ps_coscof_p2);
	y = _mm_mul_ps(y, z);
	y = _mm_mul_ps(y, z);
	__m128 tmp = _mm_mul_ps(z, *(__m128 *)_ps_0p5);
	y = _mm_sub_ps(y, tmp);
	y = _mm_add_ps(y, *(__m128 *)_ps_1);

	/* Evaluate the second polynom  (Pi/4 <= x <= 0) */

	__m128 y2 = *(__m128 *)_ps_sincof_p0;
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p1);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_add_ps(y2, *(__m128 *)_ps_sincof_p2);
	y2 = _mm_mul_ps(y2, z);
	y2 = _mm_mul_ps(y2, x);
	y2 = _mm_add_ps(y2, x);

	/* select the correct result from the two polynoms */
	xmm3 = poly_mask;
	__m128 ysin2 = _mm_and_ps(xmm3, y2);
	__m128 ysin1 = _mm_andnot_ps(xmm3, y);
	y2 = _mm_sub_ps(y2, ysin2);
	y = _mm_sub_ps(y, ysin1);

	xmm1 = _mm_add_ps(ysin1, ysin2);
	xmm2 = _mm_add_ps(y, y2);

	/* update the sign */
	*s = _mm_xor_ps(xmm1, sign_bit_sin);
	*c = _mm_xor_ps(xmm2, sign_bit_cos);
}
inline vector4 atan2(const vector4 &x, const vector4 &y)
{
	const auto zero4 = vector4(_mm_setzero_ps());
	if (((x == zero4) & (y == zero4)).move_mask() == 15)
		return zero4;

	const auto a4 = min(x, y) / max(x, y);
	const auto s4 = a4 * a4;
	const auto r4 = ((-0.0464964749f * s4 + 0.15931422f) * s4 - 0.327622764f) * s4 * a4 + a4;

	vector4 result4 = r4;
	result4.store(glm::half_pi<float>() - r4, y > x);
	result4.store(glm::pi<float>() - r4, x < zero4);
	result4.store(-1.0f * r4, y < zero4);
	return result4;
}
inline vector4 acos(const vector4 &op)
{
	return ((-0.69813170079773212f * op * op - 0.87266462599716477f) * op + 1.5707963267948966f);
} // namespace simd

static const vector4 ZERO4 = _mm_setzero_ps();
static const vector4 ONE4 = _mm_set1_ps(1.0f);
static const vector8 ONE8 = _mm256_set1_ps(1.0f);

inline glm::vec3 operator*(const matrix4 &mat, const glm::vec3 &op)
{
	const __m128 v = _mm_maskload_ps(value_ptr(op), _mm_set_epi32(~0, ~0, ~0, 0));

	__m128 v0 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
	__m128 v1 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
	__m128 v2 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));
	__m128 v3 = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3));

	__m128 m0 = _mm_mul_ps(mat.cols[0], v0);
	__m128 m1 = _mm_mul_ps(mat.cols[1], v1);
	__m128 m2 = _mm_mul_ps(mat.cols[2], v2);
	__m128 m3 = _mm_mul_ps(mat.cols[3], v3);

	__m128 a0 = _mm_add_ps(m0, m1);
	__m128 a1 = _mm_add_ps(m2, m3);
	__m128 result = _mm_add_ps(a0, a1);
	glm::vec3 res3;
	memcpy(value_ptr(res3), &result, sizeof(glm::vec3));
	return res3;
}

inline glm::vec3 operator*(const glm::vec3 &op, const matrix4 &mat)
{
	__m256 i01 = mat.cols8[0];
	__m256 i23 = mat.cols8[1];

	const __m128 v = _mm_maskload_ps(value_ptr(op), _mm_set_epi32(~0, ~0, ~0, 0));
	const __m256 v8 = _mm256_set_m128(v, v);

	__m256 m01 = _mm256_mul_ps(v8, i01);
	__m256 m23 = _mm256_mul_ps(v8, i23);

	__m128 u0 = _mm_unpacklo_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 u1 = _mm_unpackhi_ps(_mm256_extractf128_ps(m01, 0), _mm256_extractf128_ps(m01, 1));
	__m128 a0 = _mm_add_ps(u0, u1);

	__m128 u2 = _mm_unpacklo_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 u3 = _mm_unpackhi_ps(_mm256_extractf128_ps(m23, 0), _mm256_extractf128_ps(m23, 1));
	__m128 a1 = _mm_add_ps(u2, u3);

	__m128 f0 = _mm_movelh_ps(a0, a1);
	__m128 f1 = _mm_movehl_ps(a1, a0);

	__m128 result = _mm_add_ps(f0, f1);
	glm::vec3 res3;
	memcpy(value_ptr(res3), &result, sizeof(glm::vec3));
	return res3;
}
} // namespace simd
} // namespace rfw