#include "SceneAnimation.h"

#include <tiny_gltf.h>

#include "SceneObject.h"

rfw::SceneAnimation::Sampler creategLTFSampler(const tinygltf::AnimationSampler &gltfSampler,
											   const tinygltf::Model &gltfModel)
{
	rfw::SceneAnimation::Sampler sampler = {};

	if (gltfSampler.interpolation == "STEP")
		sampler.method = rfw::SceneAnimation::Sampler::STEP;
	else if (gltfSampler.interpolation == "CUBICSPLINE")
		sampler.method = rfw::SceneAnimation::Sampler::SPLINE;
	else if (gltfSampler.interpolation == "LINEAR")
		sampler.method = rfw::SceneAnimation::Sampler::LINEAR;

	// Extract animation times
	const auto &inputAccessor = gltfModel.accessors[gltfSampler.input];
	assert(inputAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);

	auto bufferView = gltfModel.bufferViews[inputAccessor.bufferView];
	auto buffer = gltfModel.buffers[bufferView.buffer];

	const float *a = (const float *)(buffer.data.data() + bufferView.byteOffset + inputAccessor.byteOffset);

	size_t count = inputAccessor.count;
	for (int i = 0; i < count; i++)
		sampler.t.push_back(a[i]);

	// extract animation keys
	auto outputAccessor = gltfModel.accessors[gltfSampler.output];
	bufferView = gltfModel.bufferViews[outputAccessor.bufferView];
	buffer = gltfModel.buffers[bufferView.buffer];

	const unsigned char *b =
		(const unsigned char *)(buffer.data.data() + bufferView.byteOffset + outputAccessor.byteOffset);
	if (outputAccessor.type == TINYGLTF_TYPE_VEC3)
	{
		// b is an array of floats (for scale or translation)
		auto f = (float *)b;
		const int N = (int)outputAccessor.count;
		for (int i = 0; i < N; i++)
			sampler.vec3_key.push_back(vec3(f[i * 3], f[i * 3 + 1], f[i * 3 + 2]));
	}
	else if (outputAccessor.type == TINYGLTF_TYPE_SCALAR)
	{
		// b can be FLOAT, BYTE, UBYTE, SHORT or USHORT... (for weights)
		std::vector<float> fdata;
		const int N = (int)outputAccessor.count;
		switch (outputAccessor.componentType)
		{
		case TINYGLTF_COMPONENT_TYPE_FLOAT:
			for (int k = 0; k < N; k++, b += 4)
				fdata.push_back(*((float *)b));
			break;
		case TINYGLTF_COMPONENT_TYPE_BYTE:
			for (int k = 0; k < N; k++, b++)
				fdata.push_back(max(*((char *)b) / 127.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			for (int k = 0; k < N; k++, b++)
				fdata.push_back(*((char *)b) / 255.0f);
			break;
		case TINYGLTF_COMPONENT_TYPE_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(max(*((short *)b) / 32767.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(*((unsigned short *)b) / 65535.0f);
			break;
		}

		for (int i = 0; i < N; i++)
			sampler.float_key.push_back(fdata[i]);
	}
	else if (outputAccessor.type == TINYGLTF_TYPE_VEC4)
	{
		// b can be FLOAT, BYTE, UBYTE, SHORT or USHORT... (for rotation)
		std::vector<float> fdata;
		const int N = (int)outputAccessor.count * 4;
		switch (outputAccessor.componentType)
		{
		case TINYGLTF_COMPONENT_TYPE_FLOAT:
			for (int k = 0; k < N; k++, b += 4)
				fdata.push_back(*((float *)b));
			break;
		case TINYGLTF_COMPONENT_TYPE_BYTE:
			for (int k = 0; k < N; k++, b++)
				fdata.push_back(max(*((char *)b) / 127.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
			for (int k = 0; k < N; k++, b++)
				fdata.push_back(*((char *)b) / 255.0f);
			break;
		case TINYGLTF_COMPONENT_TYPE_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(max(*((short *)b) / 32767.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(*((unsigned short *)b) / 65535.0f);
			break;
		}
		for (int i = 0; i < outputAccessor.count; i++)
			sampler.quat_key.push_back(quat(fdata[i * 4 + 3], fdata[i * 4], fdata[i * 4 + 1], fdata[i * 4 + 2]));
	}
	else
	{
		assert(false);
	}

	return sampler;
}

rfw::SceneAnimation::Channel creategLTFChannel(const tinygltf::AnimationChannel &gltfChannel,
											   const tinygltf::Model &gltfModel, const int nodeBase)
{
	rfw::SceneAnimation::Channel channel = {};

	channel.samplerIdx = gltfChannel.sampler;
	channel.nodeIdx = gltfChannel.target_node + nodeBase;
	if (gltfChannel.target_path == "translation")
		channel.target = rfw::SceneAnimation::Channel::TRANSLATION;
	else if (gltfChannel.target_path == "rotation")
		channel.target = rfw::SceneAnimation::Channel::ROTATION;
	else if (gltfChannel.target_path == "scale")
		channel.target = rfw::SceneAnimation::Channel::SCALE;
	else if (gltfChannel.target_path == "weights")
		channel.target = rfw::SceneAnimation::Channel::WEIGHTS;

	return channel;
}

rfw::SceneAnimation creategLTFAnim(rfw::SceneObject *object, tinygltf::Animation &gltfAnim, tinygltf::Model &gltfModel,
								   const int nodeBase)
{
	assert(object);

	rfw::SceneAnimation anim = {};

	for (size_t i = 0; i < gltfAnim.samplers.size(); i++)
		anim.samplers.push_back(creategLTFSampler(gltfAnim.samplers.at(i), gltfModel));

	for (size_t i = 0; i < gltfAnim.channels.size(); i++)
		anim.channels.push_back(creategLTFChannel(gltfAnim.channels.at(i), gltfModel, nodeBase));

	anim.object = object;

	return anim;
}

void rfw::SceneAnimation::reset()
{
	for (auto &channel : channels)
		channel.reset();
}

void rfw::SceneAnimation::update(float deltaTime)
{
	for (int i = 0; i < channels.size(); i++)
		channels[i].update(object, deltaTime, samplers[channels[i].samplerIdx]);
}

float rfw::SceneAnimation::Sampler::sampleFloat(float currentTime, int k, int i, int count) const
{
	const int keyCount = (int)t.size();
	const float animDuration = t.at(keyCount - 1);
	const float t0 = t.at(k % keyCount);
	const float t1 = t.at((k + 1) % keyCount);
	const float f = (currentTime - t0) / (t1 - t0);

	if (f <= 0)
		return float_key.at(0);

	switch (method)
	{
	case Sampler::SPLINE:
	{
		const float t = f;
		const float t2 = t * t;
		const float t3 = t2 * t;
		const float p0 = float_key.at((k * count + i) * 3 + 1);
		const float m0 = (t1 - t0) * float_key.at((k * count + i) * 3 + 2);
		const float p1 = float_key.at(((k + 1) * count + i) * 3 + 1);
		const float m1 = (t1 - t0) * float_key[((k + 1) * count + i) * 3];
		return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
	}
	case Sampler::STEP:
		return float_key.at(k);
	case Sampler::LINEAR:
	default:
		return (1.0f - f) * float_key.at(k * count + i) + f * float_key.at((k + 1.0f) * count + i);
	};
}

glm::vec3 rfw::SceneAnimation::Sampler::sampleVec3(float currentTime, int k) const
{
	const auto keyCount = t.size();
	const float t0 = t.at(k % keyCount);
	const float t1 = t.at((k + 1) % keyCount);
	const float f = (currentTime - t0) / (t1 - t0);

	return sampleFromVector(vec3_key, method, k, t0, t1, f);
}

glm::quat rfw::SceneAnimation::Sampler::sampleQuat(float currentTime, int k) const
{
	// determine interpolation parameters
	const auto keyCount = t.size();
	const float animDuration = t[keyCount - 1];
	const float t0 = t.at(k % keyCount);
	const float t1 = t.at((k + 1) % keyCount);
	const float f = (currentTime - t0) / (t1 - t0);

	const glm::quat key = sampleFromVector<glm::quat>(quat_key, method, k, t0, t1, f);
	return glm::normalize(key);
}

void rfw::SceneAnimation::Channel::reset()
{
	t = 0.0f;
	k = 0;
}

void rfw::SceneAnimation::Channel::update(rfw::SceneObject *object, const float dt, const Sampler &sampler)
{
	// Advance animation timer
	t += dt;
	auto keyCount = sampler.t.size();
	float animDuration = sampler.t.at(keyCount - 1);

	while (t > animDuration)
	{
		t -= animDuration;
		k = 0;
	}

	while (t > sampler.t.at((k + 1) % keyCount))
	{
		k++;
	}

	// Determine interpolation parameters
	const float t0 = sampler.t.at(k % keyCount);
	const float t1 = sampler.t.at((k + 1) % keyCount);
	const float f = (t - t0) / (t1 - t0);

	auto &node = object->nodes.at(nodeIdx);

	// Apply animation key
	if (target == rfw::SceneAnimation::Channel::TRANSLATION) // translation
	{
		node.translation = sampler.sampleVec3(t, k);
		node.transformed = true;
	}
	else if (target == rfw::SceneAnimation::Channel::ROTATION) // rotation
	{
		node.rotation = sampler.sampleQuat(t, k);
		node.transformed = true;
	}
	else if (target == rfw::SceneAnimation::Channel::SCALE) // scale
	{
		node.scale = sampler.sampleVec3(t, k);
		node.transformed = true;
	}
	else if (rfw::SceneAnimation::Channel::WEIGHTS) // weight
	{
		auto weightCount = node.weights.size();
		for (int i = 0; i < weightCount; i++)
			node.weights[i] = sampler.sampleFloat(t, k, i, weightCount);

		node.morphed = true;
	}
}
