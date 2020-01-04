#include "MathIncludes.h"

SIMDMat4 SIMDMat4::operator*(float op) const
{
	SIMDMat4 result;
	const __m128 op4 = _mm_set_ps1(op);

	result.cols[0] = _mm_mul_ps(cols[0], op4);
	result.cols[1] = _mm_mul_ps(cols[1], op4);
	result.cols[2] = _mm_mul_ps(cols[2], op4);
	result.cols[3] = _mm_mul_ps(cols[3], op4);

	return result;
}

SIMDMat4 SIMDMat4::operator/(float op) const
{
	SIMDMat4 result;
	const __m128 op4 = _mm_set_ps1(op);

	result.cols[0] = _mm_div_ps(cols[0], op4);
	result.cols[1] = _mm_div_ps(cols[1], op4);
	result.cols[2] = _mm_div_ps(cols[2], op4);
	result.cols[3] = _mm_div_ps(cols[3], op4);

	return result;
}

SIMDMat4 SIMDMat4::operator+(float op) const
{
	SIMDMat4 result;
	const __m128 op4 = _mm_set_ps1(op);

	result.cols[0] = _mm_add_ps(cols[0], op4);
	result.cols[1] = _mm_add_ps(cols[1], op4);
	result.cols[2] = _mm_add_ps(cols[2], op4);
	result.cols[3] = _mm_add_ps(cols[3], op4);

	return result;
}

SIMDMat4 SIMDMat4::operator-(float op) const
{
	SIMDMat4 result;
	const __m128 op4 = _mm_set_ps1(op);

	result.cols[0] = _mm_sub_ps(cols[0], op4);
	result.cols[1] = _mm_sub_ps(cols[1], op4);
	result.cols[2] = _mm_sub_ps(cols[2], op4);
	result.cols[3] = _mm_sub_ps(cols[3], op4);

	return result;
}

SIMDMat4 SIMDMat4::operator*(const SIMDMat4 &op) const
{
	SIMDMat4 result;
	glm_mat4_mul(cols, op.cols, result.cols);
	return result;
}

SIMDMat4 SIMDMat4::operator+(const SIMDMat4 &op) const
{
	SIMDMat4 result;
	glm_mat4_add(cols, op.cols, result.cols);
	return result;
}

SIMDMat4 SIMDMat4::inversed() const
{
	SIMDMat4 inv;
	glm_mat4_inverse(cols, inv.cols);
	return inv;
}

SIMDMat4 SIMDMat4::transposed() const
{
	SIMDMat4 tr;
	glm_mat4_transpose(cols, tr.cols);
	return tr;
}
