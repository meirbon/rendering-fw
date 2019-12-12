#include "Ray.h"

Ray::CameraParams::CameraParams(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height)
{
	pos_lensSize = glm::vec4(view.pos, view.aperture);
	right_spreadAngle = vec4(view.p2 - view.p1, view.spreadAngle);
	up = vec4(view.p3 - view.p1, 1.0f);
	p1 = vec4(view.p1, 1.0f);

	samplesTaken = samples;
	geometryEpsilon = 1e-5f;
	scrwidth = width;
	scrheight = height;
}

Ray Ray::generateFromView(const Ray::CameraParams &camera, int x, int y, float r0, float r1, float r2, float r3)
{
	Ray ray;
	const float blade = int(r0 * 9);
	r2 = (r2 - blade * (1.0f / 9.0f)) * 9.0f;
	float x1, y1, x2, y2;
	const float piOver4point5 = 3.14159265359f / 4.5f;
	float bladeParam = blade * piOver4point5;
	x1 = cos(bladeParam);
	y1 = sin(bladeParam);
	bladeParam = (blade + 1.0f) * piOver4point5;
	x2 = cos(bladeParam);
	y2 = sin(bladeParam);
	if ((r2 + r3) > 1.0f)
	{
		r2 = 1.0f - r2;
		r3 = 1.0f - r3;
	}
	const float xr = x1 * r2 + x2 * r3;
	const float yr = y1 * r2 + y2 * r3;
	const vec3 p1 = camera.p1;
	const vec3 right = camera.right_spreadAngle;
	const vec3 up = camera.up;

	ray.origin = vec3(camera.pos_lensSize) + camera.pos_lensSize.w * (right * xr + up * yr);
	const float u = (float(x) + r0) * (1.0f / float(camera.scrwidth));
	const float v = (float(y) + r1) * (1.0f / float(camera.scrheight));
	const vec3 pointOnPixel = p1 + u * right + v * up;
	ray.direction = normalize(pointOnPixel - ray.origin);

	return ray;
}
