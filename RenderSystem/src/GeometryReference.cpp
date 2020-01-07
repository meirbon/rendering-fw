#include "GeometryReference.h"

#include "RenderSystem.h"

namespace rfw
{

bool GeometryReference::isAnimated() const { return m_System->m_Models[m_Index]->isAnimated(); }

void GeometryReference::setAnimationTime(const float time) const { m_System->setAnimationTime(*this, time); }

const std::vector<std::pair<size_t, Mesh>> &GeometryReference::getMeshes() const { return m_System->m_Models[m_Index]->getMeshes(); }

const std::vector<simd::matrix4> &GeometryReference::getMeshMatrices() const { return m_System->m_Models[m_Index]->getMeshTransforms(); }

const std::vector<std::vector<int>> &GeometryReference::getLightIndices() const
{
	const auto lightFlags = m_System->m_Materials->getMaterialLightFlags();
	return m_System->m_Models[m_Index]->getLightIndices(lightFlags, false);
}

SceneTriangles *GeometryReference::getObject() const { return m_System->m_Models[m_Index]; }

} // namespace rfw