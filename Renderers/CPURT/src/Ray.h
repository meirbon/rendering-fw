#pragma once

#include <glm/glm.hpp>

#include <DeviceStructures.h>
#include <Camera.h>

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
