#pragma once

#include <string_view>
#include <vector>

#include "MathIncludes.h"

namespace rfw
{

class Skybox
{
  public:
	Skybox() = default;
	Skybox(std::string_view file);

	static Skybox generateTestSky();

	[[nodiscard]] const std::vector<glm::vec3> &getBuffer() const;
	[[nodiscard]] const glm::vec3 *getData() const;
	[[nodiscard]] unsigned int getWidth() const;
	[[nodiscard]] unsigned int getHeight() const;

	void load(std::string_view file);

  private:
	std::vector<glm::vec3> m_Pixels; // Sky dome pixel data
	std ::vector<float> m_Cdf;		 // CDF for importance sampling
	std::vector<float> m_Pdf;		 // PDF for importance sampling
	std::vector<float> m_ColumnCdf;  // Column CDF for importance sampling
	uint m_Width = 0, m_Height = 0;
};

} // namespace rfw