#pragma once

#include <cassert>
#include <tuple>
#include <vector>

#include "MathIncludes.h"
#include "SceneTriangles.h"

namespace rfw
{
class RenderSystem;
// Use lightweight object as a geometry reference for now, we might want to expand in the future
class GeometryReference
{
	friend class rfw::RenderSystem;

  private:
	GeometryReference(size_t index, rfw::RenderSystem &system) : m_Index(index), m_System(&system) { assert(m_System); }

  public:
	GeometryReference() = default;
	explicit operator size_t() const { return static_cast<size_t>(m_Index); }
	[[nodiscard]] size_t getIndex() const { return m_Index; }

	[[nodiscard]] bool isAnimated() const;
	void setAnimationTime(float time) const;
	[[nodiscard]] const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const;
	[[nodiscard]] const std::vector<SIMDMat4> &getMeshMatrices() const;
	[[nodiscard]] const std::vector<std::vector<int>> &getLightIndices() const;

  protected:
	[[nodiscard]] SceneTriangles *getObject() const;

  private:
	size_t m_Index; // Loaded geometry index
	rfw::RenderSystem *m_System;
};
} // namespace rfw