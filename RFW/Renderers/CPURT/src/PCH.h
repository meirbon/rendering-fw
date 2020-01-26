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

#include "BVH/AABB.h"

#include "Triangle.h"

#include "BVH/BVHNode.h"
#include "BVH/BVHTree.h"

#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"
#include "BVH/TopLevelBVH.h"

#include "Context.h"
#include "Traversal.h"

#include <bsdf/bsdf.h>