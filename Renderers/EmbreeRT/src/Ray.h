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
	static RTCRayHit8 GenerateRay8(const CameraParams &camera, const int x[8], const int y[8], rfw::utils::RandomGenerator *rng);
	static RTCRayHit16 GenerateRay16(const CameraParams &camera, const int x[16], const int y[16], rfw::utils::RandomGenerator *rng);

	template <int N> static RTCRayHitNt<N> GenerateRayN(const CameraParams &camera, int x0, int y0, int x1, int y1, rfw::utils::RandomGenerator *rng)
	{
		static_assert(N % 4 == 0, "Packet must be a multiple of 4 rays");
		assert(x1 - x0 == y1 - y0);

		RTCRayHitNt<N> query;

		int offset = 0;

		for (int y = y0; y < y1; y += 4)
		{
			for (int x = x0; x < x1; x += 4)
			{
				const int xs[4] = {x, x + 1, x, x + 1};
				const int ys[4] = {y, y, y + 1, y + 1};
				const auto packet = GenerateRay4(camera, xs, ys, rng);

				memcpy(query.hit.geomID + offset, packet.hit.geomID, 4 * sizeof(int));
				memcpy(query.hit.instID + offset, packet.hit.instID, 4 * sizeof(int));
				memcpy(query.hit.primID + offset, packet.hit.primID, 4 * sizeof(int));

				memcpy(query.ray.org_x + offset, packet.ray.org_x, 4 * sizeof(float));
				memcpy(query.ray.org_y + offset, packet.ray.org_y, 4 * sizeof(float));
				memcpy(query.ray.org_z + offset, packet.ray.org_z, 4 * sizeof(float));

				memcpy(query.ray.dir_x + offset, packet.ray.dir_x, 4 * sizeof(float));
				memcpy(query.ray.dir_y + offset, packet.ray.dir_y, 4 * sizeof(float));
				memcpy(query.ray.dir_z + offset, packet.ray.dir_z, 4 * sizeof(float));

				memcpy(query.ray.id + offset, packet.ray.id, 4 * sizeof(int));
				memcpy(query.ray.tnear + offset, packet.ray.tnear, 4 * sizeof(float));
				memcpy(query.ray.tfar + offset, packet.ray.tfar, 4 * sizeof(float));
			}
		}

		return query;
	}
};
