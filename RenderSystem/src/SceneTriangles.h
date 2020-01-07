#pragma once

#include <Structures.h>

namespace rfw
{
class RenderSystem;
class LoadException : public std::exception
{
  public:
	explicit LoadException(std::string message) : m_Message(std::move(message)) {}

	[[nodiscard]] const char *what() const noexcept override { return m_Message.c_str(); }

  private:
	std::string m_Message;
};

class SceneTriangles
{
  public:
	friend class RenderSystem;

	SceneTriangles() = default;
	virtual ~SceneTriangles() = default;

	virtual void transformTo(float timeInSeconds = 0.0f){};

	virtual const std::vector<std::pair<size_t, rfw::Mesh>> &getMeshes() const = 0;
	virtual const std::vector<simd::matrix4> &getMeshTransforms() const = 0;
	virtual const std::vector<std::vector<int>> &getLightIndices(const std::vector<bool> &matLightFlags, bool reinitialize = false) = 0;

	virtual std::vector<bool> getChangedMeshes() = 0;
	virtual std::vector<bool> getChangedMeshMatrices() = 0;

	virtual Triangle *getTriangles() = 0;
	virtual glm::vec4 *getVertices() = 0;
	virtual bool isAnimated() const { return false; }

  protected:
	virtual void prepareMeshes(RenderSystem &rs) = 0;
};
} // namespace rfw