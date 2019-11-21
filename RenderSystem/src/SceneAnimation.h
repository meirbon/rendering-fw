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

		std::vector<float> t;			 // key frame times
		std::vector<glm::vec3> vec3_key; // vec3 key frames: location or scale
		std::vector<glm::quat> quat_key; // vec4 key frames: quaternion rotations
		std::vector<float> float_key;	// float key frames: weights
		Method method;

	  private:
		template <typename T>
		static T sampleFromVector(const std::vector<T> &data, Method M, int k, const float t0, const float t1,
								  const float f)
		{
			if (f <= 0)
				return data.at(0);
			switch (M)
			{
			case (Sampler::SPLINE):
			{
				const float t = f;
				const float t2 = t * t;
				const float t3 = t2 * t;
				const auto p0 = data.at(k * 3 + 1);
				const auto m0 = (t1 - t0) * data.at(k * 3 + 2);
				const auto p1 = data.at((k + 1) * 3 + 1);
				const auto m1 = (t1 - t0) * data.at((k + 1) * 3);
				return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
			}
			case (Sampler::STEP):
				return data.at(k);
			case (Sampler::LINEAR):
			default:
				return (1.0f - f) * data.at(k) + f * data.at(k + 1);
			}
		}
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

		int samplerIdx;
		int nodeIdx;
		Target target;

		void reset();
		void update(rfw::SceneObject *object, const float t, const Sampler &sampler);

		float t = 0.0f;
		int k = 0;
	};

	SceneObject *object = nullptr;
	std::vector<Sampler> samplers;
	std::vector<Channel> channels;
	
	// TODO: Do something with these
	double ticksPerSecond = -1;
	double duration = 0;


	void reset();
	void update(float deltaTime);
};

} // namespace rfw