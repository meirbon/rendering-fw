#include "rfw.h"
#include "Internal.h"

using namespace rfw;

inline void sRGBtoLinear(unsigned char *pixels, const uint size, const uint stride)
{
	for (uint j = 0; j < size; j++)
	{
		pixels[j * stride + 0u] = (pixels[j * stride + 0u] * pixels[j * stride + 0u]) >> 8u;
		pixels[j * stride + 1u] = (pixels[j * stride + 1u] * pixels[j * stride + 1u]) >> 8u;
		pixels[j * stride + 2u] = (pixels[j * stride + 2u] * pixels[j * stride + 2u]) >> 8u;
	}
}

Texture::Texture(const std::string_view &file, uint mods)
{
	utils::Timer timer{};
	if (!utils::file::exists(file))
	{
		char buffer[1024];
		sprintf(buffer, "Could not load file: \"%s\"", file.data());
		throw std::runtime_error(buffer);
	}

	mipLevels = MIPLEVELCOUNT;
	const bool isHDR = utils::string::ends_with(file.data(), ".hdr");
	const bool shouldCache = isHDR;

	const std::string binPath = std::string(file) + ".binary";
	bool loaded = false;

	// get filetype
	FREE_IMAGE_FORMAT fif = FreeImage_GetFileType(file.data(), 0);
	if (fif == FIF_UNKNOWN)
		fif = FreeImage_GetFIFFromFilename(file.data());
	if (fif == FIF_UNKNOWN)
		FAILURE("Unsupported texture filetype: %s", file.data());

	// load image
	FIBITMAP *tmp = FreeImage_Load(fif, file.data());
	FIBITMAP *img = FreeImage_ConvertTo32Bits(tmp); // Converts 1 4 8 16 24 32 48 64 bpp to 32 bpp, fails otherwise
	if (!img)
		img = tmp;
	width = FreeImage_GetWidth(img);
	height = FreeImage_GetHeight(img);
	uint pitch = FreeImage_GetPitch(img);
	BYTE *bytes = (BYTE *)FreeImage_GetBits(img);
	uint bpp = FreeImage_GetBPP(img);
	FIBITMAP *alpha = FreeImage_GetChannel(img, FICC_ALPHA);

	if (alpha)
	{
		// set alpha rendering for this texture to true if it contains a meaningful alpha channel
		DWORD histogram[256];
		if (FreeImage_GetHistogram(alpha, histogram))
			if (histogram[0] > 0)
				flags |= HAS_ALPHA;
		FreeImage_Unload(alpha);
	}
	// iterate image pixels and write to LightHouse internal format
	if (bpp == 32) // LDR
	{
		type = UNSIGNED_INT;
		if (mods & INVERTED) // Invert image by default, free image stores the data upside down
			FreeImage_Invert(img);
		// read pixels
		texelCount = required_pixel_count(width, height, MIPLEVELCOUNT);
		udata = new uint[texelCount];
		flags |= LDR;
		for (uint y = 0; y < height; y++, bytes += pitch)
		{
			for (uint x = 0; x < width; x++)
			{
				// convert from FreeImage's 32-bit image format (usually BGRA) to 32-bit RGBA
				unsigned char *pixel = &static_cast<unsigned char *>(bytes)[x * 4];
				const uint r = pixel[FI_RGBA_RED];
				const uint g = pixel[FI_RGBA_GREEN];
				const uint b = pixel[FI_RGBA_BLUE];
				const uint a = pixel[FI_RGBA_ALPHA];
				uint rgba = (r << 0u) | (g << 8u) | (b << 16) | (a << 24);

				if (mods & FLIPPED)
					udata[((height - 1 - y) * width) + x] = rgba;
				else
					udata[(y * width) + x] = rgba;
			}
		}
		// perform sRGB -> linear conversion if requested
		if (mods & LINEARIZED)
			sRGBtoLinear((unsigned char *)udata, width * height, 4);

		// produce the MIP maps
		construct_mipmaps();
	}
	else // HDR
	{
		type = FLOAT4;
		texelCount = required_pixel_count(width, height, 1);
		fdata = new glm::vec4[texelCount]; // no MIPs for HDR for now
		flags |= HDR;
		for (uint y = 0; y < height; y++, bytes += pitch)
			for (uint x = 0; x < width; x++)
			{
				glm::vec4 rgba;
				if (bpp == 96)
					rgba = glm::vec4(((glm::vec3 *)bytes)[x], 1.0f); // 96-bit RGB, append alpha channel
				else if (bpp == 128)
					rgba = ((glm::vec4 *)bytes)[x]; // 128-bit RGBA
				(mods & FLIPPED)
					? fdata[(y * width) + x] = rgba
					: fdata[((height - 1 - y) * width) + x] = rgba; // FreeImage stores the data upside down by default
			}
	}
	// mark normal map
	if (mods & NORMAL_MAP)
		flags |= NMAP;
	// unload
	FreeImage_Unload(img);
	if (bpp == 32)
		FreeImage_Unload(tmp);

	char buffer[512];
	sprintf(buffer, "Loaded \"%s\" in %3.3f", file.data(), timer.elapsed() / 1000.0f);
	DEBUG(buffer);
}

rfw::Texture::Texture(const uint *data, uint w, uint h)
{
	type = UNSIGNED_INT;
	width = w;
	height = h;
	texelCount = required_pixel_count(w, h, MIPLEVELCOUNT);
	mipLevels = 1;
	this->udata = new uint[texelCount];
	memcpy(udata, data, width * height * sizeof(uint));
	construct_mipmaps();
}

rfw::Texture::Texture(const glm::vec4 *data, uint w, uint h)
{
	type = FLOAT4;
	width = w;
	height = h;
	texelCount = width * height;
	mipLevels = 1;
	this->fdata = new vec4[required_pixel_count(w, h, MIPLEVELCOUNT)];
	memcpy(udata, data, width * height * sizeof(vec4));
	construct_mipmaps();
}

uint rfw::Texture::sample(float x, float y)
{
	x = max(0.0f, mod(x, 1.0f));
	y = max(0.0f, mod(y, 1.0f));

	const uint u = static_cast<uint>(x * float(width - 1u));
	const uint v = static_cast<uint>(x * float(height - 1u));

	return this->udata[u + v * width];
}

void rfw::Texture::construct_mipmaps()
{
	uint *src = (uint *)this->udata;
	uint *dst = src + (width * height);

	uint pw = width;
	uint w = width >> 1u;
	uint ph = height;
	uint h = height >> 1u;

	for (uint i = 1; i < MIPLEVELCOUNT; i++)
	{
		const auto maxDst = (dst + (w * h));
		const auto maxSrc = (this->udata + texelCount);

		assert(maxDst <= maxSrc);

		for (uint y = 0; y < h; y++)
		{
			for (uint x = 0; x < w; x++)
			{
				const uint src0 = src[x * 2 + (y * 2) * pw];
				const uint src1 = src[x * 2 + 1 + (y * 2) * pw];
				const uint src2 = src[x * 2 + (y * 2 + 1) * pw];
				const uint src3 = src[x * 2 + 1 + (y * 2 + 1) * pw];
				const uint a = min(min((src0 >> 24u) & 255u, (src1 >> 24u) & 255u),
								   min((src2 >> 24u) & 255u, (src3 >> 24u) & 255u));
				const uint r =
					((src0 >> 16u) & 255u) + ((src1 >> 16u) & 255u) + ((src2 >> 16u) & 255u) + ((src3 >> 16u) & 255u);
				const uint g =
					((src0 >> 8u) & 255u) + ((src1 >> 8u) & 255u) + ((src2 >> 8u) & 255u) + ((src3 >> 8u) & 255u);
				const uint b = (src0 & 255u) + (src1 & 255u) + (src2 & 255u) + (src3 & 255u);
				dst[x + y * w] = (a << 24u) + ((r >> 2u) << 16u) + ((g >> 2u) << 8u) + (b >> 2u);
			}
		}

		src = dst;
		dst += w * h;

		pw = w;
		ph = h;
		w >>= 1u;
		h >>= 1u;
	}

	mipLevels = MIPLEVELCOUNT;
}

uint rfw::Texture::required_pixel_count(const uint width, const uint height, const uint mipLevels)
{
	auto w = width;
	auto h = height;
	uint needed = 0;

	for (uint i = 0; i < mipLevels; i++)
	{
		needed += w * h;
		w >>= 1u;
		h >>= 1u;
	}

	return needed;
}
