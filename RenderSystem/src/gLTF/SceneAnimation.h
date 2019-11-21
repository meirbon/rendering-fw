#pragma once

#include <MathIncludes.h>

#include <vector>

namespace rfw
{
class SceneObject;
class SceneAnimation
{
  public:
	struct Sampler
	{
		enum Method
		{
			LINEAR,
			SPLINE,
			STEP
		};

		float sampleFloat(float t, int k, int i, int count) const;
		glm::vec3 sampleVec3(float t, int k) const;
		glm::quat sampleQuat(float t, int k) const;

		std::vector<float> key_frames;   // key frame times
		std::vector<glm::vec3> vec3_key; // vec3 key frames: location or scale
		std::vector<glm::quat> quat_key; // vec4 key frames: quaternion rotations
		std::vector<float> float_key;	// float key frames: weights
		Method method;
	};

	struct Channel
	{
		enum Target
		{
			TRANSLATION,
			ROTATION,
			SCALE,
			WEIGHTS
		};

		int samplerIdx = -1;
		int nodeIdx = -1;
		int key = 0;
		float time = 0;
		Target target;

		void update(rfw::SceneObject *object, const float deltaTime, const Sampler &sampler);
		void setTime(rfw::SceneObject *object, const float currentTime, const Sampler &sampler);
	};

	SceneObject *object = nullptr;
	std::vector<Sampler> samplers;
	std::vector<Channel> channels;

	// TODO: Do something with these
	double ticksPerSecond = -1;
	double duration = 0;

	void update(float deltaTime);
	void setTime(float currentTime);
};

} // namespace rfw