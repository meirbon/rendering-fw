#pragma once

#include <utils.h>
#include <RenderContext.h>
#include <Structures.h>
#include <DeviceStructures.h>
#include <MathIncludes.h>
#include <ContextExport.h>

#include <immintrin.h>
#include <glm/simd/geometric.h>

#include <tbb/tbb.h>

#include <embree3/rtcore.h>
#include <embree3/rtcore_scene.h>
#include <embree3/rtcore_geometry.h>
#include <embree3/rtcore_builder.h>
#include <embree3/rtcore_device.h>
#include <embree3/rtcore_ray.h>
#include <embree3/rtcore_buffer.h>

#include "Mesh.h"
#include "Ray.h"
#include "Triangle.h"