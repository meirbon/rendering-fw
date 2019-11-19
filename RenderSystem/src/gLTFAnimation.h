#pragma once

#include <MathIncludes.h>

#include <vector>

namespace rfw
{

class gTLFObject;

class gLTFAnimation
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

		std::vector<float> t;			 // key frame times
		std::vector<glm::vec3> vec3_key; // vec3 key frames: location or scale
		std::vector<glm::quat> quat_key; // vec4 key frames: quaternion rotations
		std::vector<float> float_key;	// float key frames: weights
		Method method;
	};

	struct Channel
	{
		int samplerIdx;
		int nodeIdx;
		int target;

		void reset();
		void update(rfw::gTLFObject &object, const float t, const Sampler &sampler);

		float t = 0.0f;
		int k = 0;
	};

	std::vector<Sampler> samplers;
	std::vector<Channel> channels;

	void reset();
	void update(rfw::gTLFObject &object, float deltaTime);
};

} // namespace rfw