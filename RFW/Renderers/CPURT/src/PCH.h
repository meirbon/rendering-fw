#pragma once

#include "rfw.h"

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

#include "Ray.h"

#include "bvh/AABB.h"

#include "bvh/BVHNode.h"
#include "bvh/BVHTree.h"

#include "bvh/MBVHNode.h"
#include "bvh/MBVHTree.h"
#include "bvh/TopLevelBVH.h"

#include <bsdf/bsdf.h>

#include "Context.h"
