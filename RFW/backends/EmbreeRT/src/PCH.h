#pragma once

#include <vector>
#include <thread>
#include <iostream>
#include <memory>

#include <rfw/utils.h>
#include <rfw/context/context.h>
#include <rfw/context/context.h>
#include <rfw/context/structs.h>
#include <rfw/context/device_structs.h>
#include <rfw/math.h>
#include <rfw/context/export.h>

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