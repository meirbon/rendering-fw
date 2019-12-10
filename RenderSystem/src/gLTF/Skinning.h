#pragma once

#include <vector>
#include <string>
#include <array>

#include <MathIncludes.h>

#include <glm/gtx/matrix_major_storage.hpp>
#include <glm/simd/matrix.h>

#define ROW_MAJOR_MESH_SKIN 0

namespace rfw
{
struct SIMDMat4
{
	SIMDMat4() = default;
	SIMDMat4(glm::mat4 m) : matrix(m) {}
	SIMDMat4(glm_vec4 m[4]) { memcpy(cols, m, sizeof(cols)); }

	union {
		glm::mat4 matrix;
		glm_vec4 cols[4];
	};

	SIMDMat4 operator*(float op) const
	{
		SIMDMat4 result;
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_mul_ps(cols[0], op4);
		result.cols[1] = _mm_mul_ps(cols[1], op4);
		result.cols[2] = _mm_mul_ps(cols[2], op4);
		result.cols[3] = _mm_mul_ps(cols[3], op4);

		return result;
	}

	SIMDMat4 operator/(float op) const
	{
		SIMDMat4 result;
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_div_ps(cols[0], op4);
		result.cols[1] = _mm_div_ps(cols[1], op4);
		result.cols[2] = _mm_div_ps(cols[2], op4);
		result.cols[3] = _mm_div_ps(cols[3], op4);

		return result;
	}

	SIMDMat4 operator+(float op) const
	{
		SIMDMat4 result;
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_add_ps(cols[0], op4);
		result.cols[1] = _mm_add_ps(cols[1], op4);
		result.cols[2] = _mm_add_ps(cols[2], op4);
		result.cols[3] = _mm_add_ps(cols[3], op4);

		return result;
	}

	SIMDMat4 operator-(float op) const
	{
		SIMDMat4 result;
		const __m128 op4 = _mm_set_ps1(op);

		result.cols[0] = _mm_sub_ps(cols[0], op4);
		result.cols[1] = _mm_sub_ps(cols[1], op4);
		result.cols[2] = _mm_sub_ps(cols[2], op4);
		result.cols[3] = _mm_sub_ps(cols[3], op4);

		return result;
	}

	SIMDMat4 operator*(const SIMDMat4 &op) const
	{
		SIMDMat4 result;
		glm_mat4_mul(cols, op.cols, result.cols);
		return result;
	}

	SIMDMat4 operator+(const SIMDMat4 &op) const
	{
		SIMDMat4 result;
		glm_mat4_add(cols, op.cols, result.cols);
		return result;
	}

	SIMDMat4 inversed() const
	{
		SIMDMat4 inv;
		glm_mat4_inverse(cols, inv.cols);
		return inv;
	}

	SIMDMat4 transposed() const
	{
		SIMDMat4 tr;
		glm_mat4_transpose(cols, tr.cols);
		return tr;
	}
};

class MeshSkin
{
  public:
	std::string name;
	std::vector<int> jointNodes;

	std::vector<SIMDMat4> inverseBindMatrices;
	std::vector<SIMDMat4> jointMatrices;
};

class MeshBone
{
  public:
	std::string name;
	unsigned int nodeIndex;

	std::vector<unsigned short> vertexIDs;
	std::vector<float> vertexWeights;
	glm::mat4 offsetMatrix;
};

} // namespace rfw