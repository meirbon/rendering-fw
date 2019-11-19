#pragma once

#include <Structures.h>

namespace rfw
{
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
	SceneTriangles() = default;
	virtual ~SceneTriangles() = default;

	virtual void transformTo(float timeInSeconds = 0.0f){};
	virtual rfw::Mesh getMesh() const = 0;
	virtual std::vector<uint> getLightIndices(const std::vector<bool> &matLightFlags) const = 0;
	virtual Triangle *getTriangles() = 0;
	virtual glm::vec4 *getVertices() = 0;
	virtual bool isAnimated() const { return false; }
	virtual uint getAnimationCount() const { return 0; }
	virtual void setAnimation(uint index){};
	virtual uint getMaterialForPrim(uint primitiveIdx) const = 0;
};
} // namespace rfw