#pragma once

#include <cassert>

#include "MathIncludes.h"

namespace rfw
{
class RenderSystem;
class LightReference
{
	friend class RenderSystem;

  public:
	enum LightType
	{
		AREA = 0,
		POINT = 1,
		SPOT = 2,
		DIRECTIONAL = 3,
		UNDEFINED = 4
	};

	LightReference() : index(0), type(UNDEFINED), m_System(nullptr){};
	LightReference(size_t index, LightType type, rfw::RenderSystem &system);
	LightReference(const LightReference &other);

	size_t index = 0;
	LightType type = UNDEFINED;

	operator size_t() const { return index; }

  private:
	rfw::RenderSystem *m_System;
};

} // namespace rfw