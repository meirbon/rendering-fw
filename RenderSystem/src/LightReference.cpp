#include "LightReference.h"
#include "RenderSystem.h"

namespace rfw
{

LightReference::LightReference(size_t idx, LightReference::LightType t, RenderSystem &system) : index(idx), m_System(&system), type(t) { assert(m_System); }

// Changing const members probably isn't best practice, but it prevents users of the api to change.
LightReference::LightReference(const LightReference &other) = default;

} // namespace rfw