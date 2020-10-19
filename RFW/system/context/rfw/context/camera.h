#pragma once

#include <glm/glm.hpp>
#include <glm/ext.hpp>

#include <string_view>

#include <rfw/context/device_structs.h>

#include <rfw/utils/serializable.h>

namespace rfw
{

class Camera
{
  public:
	static const float DEFAULT_BRIGHTNESS;
	static const float DEFAULT_CONTRAST;
	static const glm::vec3 DEFAULT_POSITION;
	static const glm::vec3 DEFAULT_DIRECTION;

	// constructor / destructor
	Camera() = default;

	// data members
	glm::vec3 position = DEFAULT_POSITION;	 // position of the centre of the lens
	glm::vec3 direction = DEFAULT_DIRECTION; // camera target

	float focalDistance = 5.0f;				  // distance of the focal plane
	float aperture = 0.0001f;				  // aperture size
	float brightness = DEFAULT_BRIGHTNESS;	  // combined with contrast:
	float contrast = DEFAULT_CONTRAST;		  // pragmatic representation of exposure
	float FOV = 40.0f;						  // field of view, in degrees
	float aspectRatio = 1.0f;				  // image plane aspect ratio
	float clampValue = 10.0f;				  // firefly clamping
	glm::ivec2 pixelCount = glm::ivec2(1, 1); // actual pixel count; needed for pixel spread angle

	void reset();

	[[nodiscard]] rfw::CameraView get_view() const;
	[[nodiscard]] glm::mat4 get_matrix(float near = 0.1f, float far = 1e10f) const;
	void resize(int w, int h);
	void look_at(const glm::vec3 &O, const glm::vec3 &T); // position the camera at O, looking at T
	void translate_relative(const glm::vec3 &T);		  // move camera relative to orientation
	void translate_target(const glm::vec3 &T);			  // move camera target; used for rotating camera

	void serialize(std::string_view file) const;
	[[nodiscard]] rfw::utils::serializable<rfw::Camera, 1> serialize() const;
	static rfw::Camera deserialize(std::string_view file);

  private:
	void calculate_matrix(glm::vec3 &x, glm::vec3 &y, glm::vec3 &z) const;
};

} // namespace rfw
