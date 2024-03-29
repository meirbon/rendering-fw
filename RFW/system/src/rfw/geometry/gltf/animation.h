#pragma once

#include <rfw/math.h>

#include <vector>

namespace rfw::geometry::gltf
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

		std::vector<float> key_frames;	 // key frame times
		std::vector<glm::vec3> vec3_key; // vec3 key frames: location or scale
		std::vector<glm::quat> quat_key; // vec4 key frames: quaternion rotations
		std::vector<float> float_key;	 // float key frames: weights
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

		std::vector<int> samplerIDs;
		std::vector<Target> targets;

		int nodeIdx = -1;
		int key = 0;
		float time = 0;

		void update(SceneObject *object, const float deltaTime, const std::vector<Sampler> &sampler);
		void setTime(SceneObject *object, const float currentTime, const std::vector<Sampler> &sampler);
	};

	SceneObject *object = nullptr;
	std::vector<Sampler> samplers;
	std::vector<Channel> channels;

	// TODO: Do something with these
	double ticksPerSecond = -1.0;
	double duration = 0.0;

	void update(float deltaTime);
	void setTime(float currentTime);
};

} // namespace rfw