#pragma once

#include <glm/glm.hpp>

#include <DeviceStructures.h>
#include <Camera.h>

#include <utils/RandomGenerator.h>

namespace cpurt
{

template <int N> struct RayPacket
{
	RayPacket()
	{
		static_assert(N % 4 == 0, "Ray packet must consist of a multiple of 4 rays.");

		memset(origin_x, 0, sizeof(origin_x));
		memset(origin_y, 0, sizeof(origin_y));
		memset(origin_z, 0, sizeof(origin_z));

		memset(direction_x, 0, sizeof(direction_x));
		memset(direction_y, 0, sizeof(direction_y));
		memset(direction_z, 0, sizeof(direction_z));

		memset(pixelID, -1, sizeof(pixelID));
		memset(primID, -1, sizeof(primID));
		memset(instID, -1, sizeof(instID));
		for (int i = 0; i < N; i++)
			t[i] = 1e34f;
	}

	union {
		float origin_x[N];
		__m128 origin_x4[N / 4];
	};
	union {
		float origin_y[N];
		__m128 origin_y4[N / 4];
	};
	union {
		float origin_z[N];
		__m128 origin_z4[N / 4];
	};

	union {
		float direction_x[N];
		__m128 direction_x4[N / 4];
	};
	union {
		float direction_y[N];
		__m128 direction_y4[N / 4];
	};
	union {
		float direction_z[N];
		__m128 direction_z4[N / 4];
	};

	union {
		int pixelID[N];
		__m128 pixelID4[N / 4];
	};
	union {
		int primID[N];
		__m128i primID4[N / 4];
	};
	union {
		int instID[N];
		__m128i instID4[N / 4];
	};

	union {
		float t[N];
		__m128 t4[N / 4];
	};
};

using RayPacket4 = RayPacket<4>;
using RayPacket8 = RayPacket<8>;

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
};
} // namespace cpurt