#pragma once

#include <glm/glm.hpp>

#include <DeviceStructures.h>
#include <Camera.h>

#include <embree3/rtcore_ray.h>

#include <utils/RandomGenerator.h>

struct Ray
{
	struct CameraParams
	{
		CameraParams(const rfw::CameraView &view, uint samples, float epsilon, uint width, uint height);
		glm::vec4 pos_lensSize;
		glm::vec4 right_spreadAngle;
		glm::vec4 up;
		glm::vec4 p1;

		int samplesTaken;
		float geometryEpsilon;
		int scrwidth;
		int scrheight;
	};

	glm::vec3 origin = glm::vec3(0.0f);
	float t = 1e34f;

	glm::vec3 direction = glm::vec3(0, 0, 1);
	int primIdx = -1;

	[[nodiscard]] bool isValid() const { return primIdx >= 0; }

	void reset()
	{
		t = 1e34f;
		primIdx = -1;
	}

	static Ray generateFromView(const CameraParams &camera, int x, int y, float r0, float r1, float r2, float r3);

	static RTCRayHit4 GenerateRay4(const CameraParams &camera, const int x[4], const int y[4], rfw::utils::RandomGenerator *rng);
	static RTCRayHit8 GenerateRay8(const CameraParams &camera, std::array<std::pair<int, int>, 8> pixels, rfw::utils::RandomGenerator *rng);
	static RTCRayHit16 GenerateRay16(const CameraParams &camera, std::array<std::pair<int, int>, 16> pixels, rfw::utils::RandomGenerator *rng);

	template <int N> static RTCRayHitNt<N> GenerateRayN(const CameraParams &camera, std::pair<int, int> *pixels, rfw::utils::RandomGenerator *rng)
	{
		RTCRayHitNt<N> query;

		// TODO

		return query;
	}
};
