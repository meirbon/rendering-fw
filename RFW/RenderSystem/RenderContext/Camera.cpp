#include "Camera.h"

#include <utils/File.h>

using namespace rfw;
using namespace glm;

const float Camera::DEFAULT_BRIGHTNESS = 0.05f;
const float Camera::DEFAULT_CONTRAST = 1.f;
const glm::vec3 Camera::DEFAULT_POSITION = glm::vec3(0.f);
const glm::vec3 Camera::DEFAULT_DIRECTION = glm::vec3(0.f, 0.f, 1.f);

void rfw::Camera::reset()
{
	brightness = DEFAULT_BRIGHTNESS;
	contrast = DEFAULT_CONTRAST;
	position = DEFAULT_POSITION;
	direction = DEFAULT_DIRECTION;
}

void Camera::look_at(const glm::vec3 &O, const glm::vec3 &T)
{
	position = O;
	direction = normalize(T - O);
}

void Camera::translate_relative(const glm::vec3 &T)
{
	glm::vec3 right, up, forward;
	calculate_matrix(right, up, forward);
	glm::vec3 delta = T.x * right + T.y * up + T.z * forward;
	position += delta;
}

void Camera::translate_target(const glm::vec3 &T)
{
	glm::vec3 right, up, forward;
	calculate_matrix(right, up, forward);
	direction = normalize(direction + T.x * right + T.y * up + T.z * forward);
}

void rfw::Camera::serialize(std::string_view file) const
{
	const auto data = serialize();
	const auto byteData = data.serialize();
	rfw::utils::file::write(file, byteData);
}

rfw::utils::Serializable<rfw::Camera, 1> rfw::Camera::serialize() const { return rfw::utils::Serializable<rfw::Camera, 1>(*this); }

rfw::Camera rfw::Camera::deserialize(std::string_view file)
{
	if (!rfw::utils::file::exists(file))
	{
		WARNING("Camera file \"%s\" does not exist.", file.data());
		return rfw::Camera();
	}

	try
	{
		const auto data = rfw::utils::Serializable<rfw::Camera, 1>::deserialize(rfw::utils::file::read_binary(file));
		return *data.get_data();
	}
	catch (const std::exception &e)
	{
		WARNING(e.what());
		return {};
	}
}

rfw::CameraView Camera::get_view() const
{
	rfw::CameraView view;
	glm::vec3 right, up, forward;
	calculate_matrix(right, up, forward);
	view.pos = position;
	view.spreadAngle = (FOV * glm::pi<float>() / 180) / float(pixelCount.y);
	const float screenSize = tan(FOV / 2.0f / (180.0f / glm::pi<float>()));
	const glm::vec3 Center = view.pos + focalDistance * forward;
	view.p1 = Center - screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p2 = Center + screenSize * right * focalDistance * aspectRatio + screenSize * focalDistance * up;
	view.p3 = Center - screenSize * right * focalDistance * aspectRatio - screenSize * focalDistance * up;
	view.aperture = aperture;
	return view;
}

mat4 Camera::get_matrix(const float nearPlane, const float farPlane) const
{
	const vec3 up = vec3(0, 1, 0);
	const vec3 forward = -direction;

	const float fovDist = tanf(glm::radians(FOV * 0.5f));

	const mat4 flip = scale(mat4(1.0f), vec3(-1));
	const mat4 projection = perspective(radians(FOV), aspectRatio, nearPlane, farPlane);
	const mat4 view = glm::lookAt(position, position + forward * fovDist, up);
	return projection * flip * view;
}

void Camera::resize(int w, int h)
{
	aspectRatio = float(w) / float(h);
	pixelCount = ivec2(w, h);
}

void Camera::calculate_matrix(glm::vec3 &x, glm::vec3 &y, glm::vec3 &z) const
{
	y = glm::vec3(0.0f, 1.0f, 0.0f);
	z = direction; // assumed to be normalized at all times
	x = normalize(cross(z, y));
	y = cross(x, z);
}

// EOF