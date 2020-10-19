#include "rfw.h"

namespace rfw
{

light_ref::light_ref(size_t idx, light_ref::LightType t, system &system) : index(idx), m_System(&system), type(t) { assert(m_System); }

// Changing const members probably isn't best practice, but it prevents users of the api to change.
light_ref::light_ref(const light_ref &other) = default;

} // namespace rfw