//
// Created by Mèir Noordermeer on 2019-08-20.
//

#ifndef RENDERING_FW_SRC_TEXTURE_HPP
#define RENDERING_FW_SRC_TEXTURE_HPP

#include <string>
#include <vector>

#include <MathIncludes.h>

namespace rfw
{

struct Texture
{
	enum Type
	{
		FLOAT4,
		UNSIGNED_INT
	};
	enum Flags
	{
		INVERTED = 1,
		LINEARIZED = 2,
		FLIPPED = 4,
		NORMAL_MAP = 8
	};
	enum Properties
	{
		HAS_ALPHA = 1,
		LDR = 2,
		HDR = 4,
		NMAP = 8,
	};
	Texture() = default;
	explicit Texture(const std::string_view &file, uint flags = 0);

	uint sample(float x, float y);

	void constructMipMaps();
	static uint requiredNumberOfPixels(uint width, uint height, uint mipLevels);

	Type type;
	uint texelCount;
	uint width, height;
	uint mipLevels;
	uint flags = 0;
	union {
		glm::vec4 *fdata;
		uint *udata;
	};
};
} // namespace rfw

#endif // RENDERING_FW_SRC_TEXTURE_HPP
