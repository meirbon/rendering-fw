#pragma once

#include <string_view>
#include <vector>
#include <string>

#include "MathIncludes.h"

namespace rfw
{

class Skybox
{
  public:
	Skybox() = default;
	Skybox(std::string_view file);
	Skybox(rfw::utils::ArrayProxy<vec3> pixels, int width, int height);
	Skybox(rfw::utils::ArrayProxy<vec4> pixels, int width, int height);

	static Skybox generate_test_sky();

	[[nodiscard]] const std::vector<glm::vec3> &get_buffer() const;
	[[nodiscard]] const glm::vec3 *get_data() const;
	[[nodiscard]] unsigned int get_width() const;
	[[nodiscard]] unsigned int get_height() const;

	void load(std::string_view file);
	void set(rfw::utils::ArrayProxy<vec3> pixels, int width, int height);
	void set(rfw::utils::ArrayProxy<vec4> pixels, int width, int height);

	[[nodiscard]] const std::string &get_source() const { return m_File; }

  private:
	std::string m_File = "";
	std::vector<glm::vec3> m_Pixels; // Sky dome pixel data
	std ::vector<float> m_Cdf;		 // CDF for importance sampling
	std::vector<float> m_Pdf;		 // PDF for importance sampling
	std::vector<float> m_ColumnCdf;	 // Column CDF for importance sampling
	uint m_Width = 0, m_Height = 0;
};

} // namespace rfw