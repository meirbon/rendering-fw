#include "Skybox.h"

#include <FreeImage.h>

#include "Settings.h"
#include "utils/String.h"
#include "utils/Timer.h"
#include "utils/File.h"
#include "utils/Serializable.h"

#define SKYCDF(x, y) cdf[RadicalInverse8bit(y) + x * (IBLHEIGHT + 1)] // columns stored sequentially for better cache coherence
#define COLCDF(x) columncdf[RadicalInverse9bit(x)]

inline static int radicalInverse8bit(const int v)
{
	int x = ((v & 0xaa) >> 1) | ((v & 0x55) << 1);
	x = ((x & 0xcc) >> 2) | ((x & 0x33) << 2);
	x = ((x & 0xf0) >> 4) | ((x & 0x0f) << 4);
	return x + (1 - (v >> 8)); // so 256 = 0
}

inline static int radicalInverse9bit(const int v)
{
	int x = ((v & 0xaa) >> 1) | ((v & 0x55) << 1);
	x = ((x & 0xcc) >> 2) | ((x & 0x33) << 2);
	x = ((x & 0xf0) >> 4) | ((x & 0x0f) << 4);
	return ((x << 1) | ((v >> 8) & 1)) + (1 - (v >> 9)); // so 512 = 0
}

rfw::Skybox::Skybox(std::string_view file) { load(file); }

rfw::Skybox rfw::Skybox::generateTestSky()
{
	Skybox skybox;
	// red / green / blue test environment
	skybox.m_Width = 5120;
	skybox.m_Height = 2560;
	skybox.m_Pixels = std::vector<glm::vec3>(skybox.m_Width * skybox.m_Height, vec3(0.0f));
	for (auto x = 0; x < 5120; x++)
		for (int y = 0; y < 2560; y++)
			skybox.m_Pixels[x + y * 5120] = vec3(0.1f, 0.1f, 0.1f);
	for (auto x = 0; x < 200; x++)
		for (auto y = 900; y < 1100; y++)
			skybox.m_Pixels[x + y * 5120] = vec3(10, 0, 0);
	for (auto x = 2000; x < 2200; x++)
		for (auto y = 900; y < 1100; y++)
			skybox.m_Pixels[x + y * 5120] = vec3(0, 10, 0);
	for (auto x = 4000; x < 4200; x++)
		for (auto y = 900; y < 1100; y++)
			skybox.m_Pixels[x + y * 5120] = vec3(0, 0, 10);

	return skybox;
}

const std::vector<glm::vec3> &rfw::Skybox::getBuffer() const { return m_Pixels; }

const glm::vec3 *rfw::Skybox::getData() const { return m_Pixels.data(); }

unsigned rfw::Skybox::getWidth() const { return m_Width; }

unsigned rfw::Skybox::getHeight() const { return m_Height; }

void rfw::Skybox::load(std::string_view file)
{
	if (utils::file::exists(file))
	{
		char buffer[1024];
		sprintf(buffer, "File \"%s\" does not exist.", file.data());
	}

	m_File = std::string(file.data());

	m_Width = 0;
	m_Height = 0;

	m_Pdf.resize(IBL_WIDTH * IBL_HEIGHT);
	m_Cdf.resize(IBL_WIDTH * (IBL_HEIGHT + 1));
	m_ColumnCdf.resize(IBL_WIDTH + 1);

	const auto filename = file.data();
	const bool isHDR = utils::string::ends_with(filename, ".hdr");
	const std::string binaryPath = std::string(filename) + ".bin";

	utils::Timer timer{};
	if (isHDR)
	{
		bool loaded = false;
#if CACHE_SKYBOX
		if (utils::file::exists(binaryPath))
		{
			const auto buffer = utils::file::read_binary(binaryPath);

			auto image = utils::Serializable<glm::vec3, 2>::deserialize(buffer);
			const std::array<unsigned int, 2> dims = image.getDimensions();

			m_Width = dims[0];
			m_Height = dims[1];

			if (m_Width == 0 || m_Height == 0 || buffer.empty())
				loaded = false;
			else
			{
				m_Pixels.resize(m_Width * m_Height);
				memcpy(m_Pixels.data(), image.getData(), m_Width * m_Height * sizeof(glm::vec3));
			}
			loaded = true;
		}
#endif

		if (!loaded)
		{
			FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
			fif = FreeImage_GetFileType(filename, 0);
			if (fif == FIF_UNKNOWN)
				fif = FreeImage_GetFIFFromFilename(filename);

			FIBITMAP *tmp = FreeImage_Load(fif, filename);
			m_Width = FreeImage_GetWidth(tmp);
			m_Height = FreeImage_GetHeight(tmp);

			FIBITMAP *dib = FreeImage_ConvertToRGBAF(tmp);
			m_Pixels.resize(m_Width * m_Height);

			for (uint y = 0; y < m_Height; y++)
			{
				const FIRGBAF *bits = (FIRGBAF *)FreeImage_GetScanLine(dib, y);
				for (uint x = 0; x < m_Width; x++)
				{
					m_Pixels.at(((m_Height - 1 - y) * m_Width) + x) = vec3(bits[x].red, bits[x].green, bits[x].blue);
				}
			}

			FreeImage_Unload(tmp);
			FreeImage_Unload(dib);

#if CACHE_SKYBOX
			utils::Serializable<glm::vec3, 2> cacheableImage(m_Pixels, {m_Width, m_Height});
			utils::file::write(binaryPath, cacheableImage.serialize());
#endif
		}
	}
	else
	{
		FREE_IMAGE_FORMAT fif = FIF_UNKNOWN;
		fif = FreeImage_GetFileType(filename, 0);
		if (fif == FIF_UNKNOWN)
			fif = FreeImage_GetFIFFromFilename(filename);

		FIBITMAP *tmp = FreeImage_Load(fif, filename);
		m_Width = FreeImage_GetWidth(tmp);
		m_Height = FreeImage_GetHeight(tmp);

		FIBITMAP *dib = FreeImage_ConvertToRGBAF(tmp);
		m_Pixels.resize(m_Width * m_Height);

		for (uint y = 0; y < m_Height; y++)
		{
			const uint yIdx = y * m_Width;
			const FIRGBAF *bits = (FIRGBAF *)FreeImage_GetScanLine(dib, y);
			for (uint x = 0; x < m_Width; x++)
				m_Pixels.at(x + yIdx) = vec3(bits[x].red, bits[x].green, bits[x].blue);
		}

		FreeImage_Unload(tmp);
		FreeImage_Unload(dib);
	}

	DEBUG("Loaded skybox \"%s\" in %3.3f seconds", filename, timer.elapsed() / 1000.0f);
}
