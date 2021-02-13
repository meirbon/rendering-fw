#pragma once

#include <immintrin.h>

#include <atomic>
#include <vector>
#include <thread>
#include <mutex>
#include <future>
#include <tuple>
#include <optional>

#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/simd/geometric.h>

#include <rfw/context/context.h>
#include <rfw/context/export.h>

#include <rfw/utils/gl/check.h>
#include <rfw/math.h>

#include <tbb/tbb.h>

#include <rfw/utils.h>

#include "Ray.h"

#include <bvh/AABB.h>

#include "Triangle.h"

#include <bvh/bvh_node.h>
#include <bvh/bvh_tree.h>

#include <bvh/mbvh_node.h>
#include <bvh/mbvh_tree.h>
#include <bvh/top_level_bvh.h>

#include "Context.h"