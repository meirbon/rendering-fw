#include "gLTFAnimation.h"

#include <tiny_gltf.h>

#include "gLTFObject.h"

rfw::gLTFAnimation::Sampler creategLTFSampler(const tinygltf::AnimationSampler &gltfSampler,
											  const tinygltf::Model &gltfModel)
{
	rfw::gLTFAnimation::Sampler sampler = {};

	if (gltfSampler.interpolation == "STEP")
		sampler.method = rfw::gLTFAnimation::Sampler::STEP;
	else if (gltfSampler.interpolation == "CUBICSPLINE")
		sampler.method = rfw::gLTFAnimation::Sampler::SPLINE;
	else if (gltfSampler.interpolation == "LINEAR")
		sampler.method = rfw::gLTFAnimation::Sampler::LINEAR;

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
				fdata.push_back(max(*((char *)b) / 32767.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(*((char *)b) / 65535.0f);
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
				fdata.push_back(max(*((char *)b) / 32767.0f, -1.0f));
			break;
		case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
			for (int k = 0; k < N; k++, b += 2)
				fdata.push_back(*((char *)b) / 65535.0f);
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

rfw::gLTFAnimation::Channel creategLTFChannel(const tinygltf::AnimationChannel &gltfChannel,
											  const tinygltf::Model &gltfModel, const int nodeBase)
{
	rfw::gLTFAnimation::Channel channel = {};

	channel.samplerIdx = gltfChannel.sampler;
	channel.nodeIdx = gltfChannel.target_node + nodeBase;
	if (gltfChannel.target_path.compare("translation") == 0)
		channel.target = 0;
	if (gltfChannel.target_path.compare("rotation") == 0)
		channel.target = 1;
	if (gltfChannel.target_path.compare("scale") == 0)
		channel.target = 2;
	if (gltfChannel.target_path.compare("weights") == 0)
		channel.target = 3;

	return channel;
}

rfw::gLTFAnimation creategLTFAnim(tinygltf::Animation &gltfAnim, tinygltf::Model &gltfModel, const int nodeBase)
{
	rfw::gLTFAnimation anim = {};

	for (size_t i = 0; i < gltfAnim.samplers.size(); i++)
		anim.samplers.push_back(creategLTFSampler(gltfAnim.samplers[i], gltfModel));
	for (size_t i = 0; i < gltfAnim.channels.size(); i++)
		anim.channels.push_back(creategLTFChannel(gltfAnim.channels[i], gltfModel, nodeBase));

	return anim;
}

void rfw::gLTFAnimation::reset()
{
	for (auto &channel : channels)
		channel.reset();
}

void rfw::gLTFAnimation::update(rfw::gTLFObject &object, float deltaTime)
{
	for (int i = 0; i < channels.size(); i++)
		channels[i].update(object, deltaTime, samplers[channels[i].samplerIdx]);
}

float rfw::gLTFAnimation::Sampler::sampleFloat(float currentTime, int k, int i, int count) const
{
	const int keyCount = (int)t.size();
	const float animDuration = t[keyCount - 1];
	const float t0 = t[k % keyCount], t1 = t[(k + 1) % keyCount];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	if (f <= 0)
		return float_key.at(0);

	switch (method)
	{
	case Sampler::SPLINE:
	{
		const float t = f, t2 = t * t, t3 = t2 * t;
		const float p0 = float_key[(k * count + i) * 3 + 1];
		const float m0 = (t1 - t0) * float_key[(k * count + i) * 3 + 2];
		const float p1 = float_key[((k + 1) * count + i) * 3 + 1];
		const float m1 = (t1 - t0) * float_key[((k + 1) * count + i) * 3];
		return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
	}
	case Sampler::STEP:
		return float_key[k];
	case Sampler::LINEAR:
	default:
		return (1.0f - f) * float_key[k * count + i] + f * float_key[(k + 1.0f) * count + i];
	};
}

glm::vec3 rfw::gLTFAnimation::Sampler::sampleVec3(float currentTime, int k) const
{
	const int keyCount = (int)t.size();
	const float animDuration = t[keyCount - 1];
	const float t0 = t[k % keyCount], t1 = t[(k + 1) % keyCount];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	if (f <= 0)
		return vec3_key[0];
	switch (method)
	{
	case Sampler::SPLINE:
	{
		const float t = f, t2 = t * t, t3 = t2 * t;
		const auto p0 = vec3_key[k * 3 + 1];
		const auto m0 = (t1 - t0) * vec3_key[k * 3 + 2];
		const auto p1 = vec3_key[(k + 1) * 3 + 1];
		const auto m1 = (t1 - t0) * vec3_key[(k + 1) * 3];
		return m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
	}
	case Sampler::STEP:
		return vec3_key[k];
	case Sampler::LINEAR:
	default:
		return (1.0f - f) * vec3_key[k] + f * vec3_key[k + 1.0f];
	};
}

glm::quat rfw::gLTFAnimation::Sampler::sampleQuat(float currentTime, int k) const
{
	// determine interpolation parameters
	const int keyCount = (int)t.size();
	const float animDuration = t[keyCount - 1];
	const float t0 = t[k % keyCount], t1 = t[(k + 1) % keyCount];
	const float f = (currentTime - t0) / (t1 - t0);
	// sample
	quat key;
	if (f <= 0)
		key = quat_key[0];
	else
		switch (method)
		{
		case Sampler::SPLINE:
		{
			const float t = f, t2 = t * t, t3 = t2 * t;
			const auto p0 = quat_key[k * 3 + 1];
			const auto m0 = quat_key[k * 3 + 2] * (t1 - t0);
			const auto p1 = quat_key[(k + 1) * 3 + 1];
			const auto m1 = quat_key[(k + 1) * 3] * (t1 - t0);
			key = m0 * (t3 - 2 * t2 + t) + p0 * (2 * t3 - 3 * t2 + 1) + p1 * (-2 * t3 + 3 * t2) + m1 * (t3 - t2);
			break;
		}
		case Sampler::STEP:
		{
			key = quat_key[k];
			break;
		default:
			// key = quat::slerp( vec4Key[k], vec4Key[k + 1], f );
			key = (quat_key[k] * (1 - f)) + (quat_key[k + 1] * f);
			break;
		}
		};

	return normalize(key);
}

void rfw::gLTFAnimation::Channel::reset()
{
	t = 0.0f;
	k = 0;
}

void rfw::gLTFAnimation::Channel::update(rfw::gTLFObject &object, const float dt, const Sampler &sampler)
{
	// Advance animation timer
	t += dt;
	auto keyCount = sampler.t.size();
	float animDuration = sampler.t.at(keyCount - 1);
	while (t > animDuration)
		t -= animDuration, k = 0;
	while (t > sampler.t[(k + 1) % keyCount])
		k++;

	// Determine interpolation parameters
	float t0 = sampler.t[k % keyCount];
	float t1 = sampler.t[(k + 1) % keyCount];
	float f = (t - t0) / (t1 - t0);

	auto &node = object.m_Nodes.at(nodeIdx);

	// Apply animation key
	if (target == 0) // translation
	{
		node.translation = sampler.sampleVec3(t, k);
		node.transformed = true;
	}
	else if (target == 1) // rotation
	{
		node.rotation = sampler.sampleQuat(t, k);
		node.transformed = true;
	}
	else if (target == 2) // scale
	{
		node.scale = sampler.sampleVec3(t, k);
		node.transformed = true;
	}
	else // target == 3, weight
	{
		auto weightCount = node.weights.size();
		for (int i = 0; i < weightCount; i++)
		{
			node.weights[i] = sampler.sampleFloat(t, k, i, weightCount);
		}

		node.morphed = true;
	}
}
