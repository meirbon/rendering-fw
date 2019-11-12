#pragma once

#include <MathIncludes.h>

#include <DeviceStructures.h>

#include <utils/Serializable.h>

namespace rfw
{

class Camera
{
  public:
	// constructor / destructor
	Camera() = default;

	// data members
	glm::vec3 position = glm::vec3(0.0f);			   // position of the centre of the lens
	glm::vec3 direction = glm::vec3(0.0f, 0.0f, 1.0f); // camera target

	float focalDistance = 5.0f;				  // distance of the focal plane
	float aperture = 0.0001f;				  // aperture size
	float brightness = 0.0f;				  // combined with contrast:
	float contrast = 0.0f;					  // pragmatic representation of exposure
	float FOV = 40.0f;						  // field of view, in degrees
	float aspectRatio = 1.0f;				  // image plane aspect ratio
	float clampValue = 10.0f;				  // firefly clamping
	glm::ivec2 pixelCount = glm::ivec2(1, 1); // actual pixel count; needed for pixel spread angle

	[[nodiscard]] rfw::CameraView getView() const;
	[[nodiscard]] mat4 getMatrix() const;
	void resize(int w, int h);
	void lookAt(const glm::vec3 &O, const glm::vec3 &T); // position the camera at O, looking at T
	void translateRelative(const glm::vec3 &T);			 // move camera relative to orientation
	void translateTarget(const glm::vec3 &T);			 // move camera target; used for rotating camera


	void serialize(std::string_view file) const;
	rfw::utils::Serializable<rfw::Camera, 1> serialize() const;
	static rfw::Camera deserialize(std::string_view file);

  private:
	void calculateMatrix(glm::vec3 &x, glm::vec3 &y, glm::vec3 &z) const;
};

} // namespace rfw
