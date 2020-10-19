#pragma once

#include <cassert>

#include <rfw_math.h>

namespace rfw
{
class system;
class light_ref
{
	friend class system;

  public:
	enum LightType
	{
		AREA = 0,
		POINT = 1,
		SPOT = 2,
		DIRECTIONAL = 3,
		UNDEFINED = 4
	};

	light_ref() : index(0), type(UNDEFINED), m_System(nullptr){};
	light_ref(size_t index, LightType type, rfw::system &system);
	light_ref(const light_ref &other);

	size_t index = 0;
	LightType type = UNDEFINED;

	operator size_t() const { return index; }

  private:
	rfw::system *m_System;
};

} // namespace rfw