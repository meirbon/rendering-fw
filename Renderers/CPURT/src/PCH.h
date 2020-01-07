#pragma once

#include "rfw.h"

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
#include <immintrin.h>

#include "Ray.h"
#include "Mesh.h"
#include "Triangle.h"

#include "BVH/AABB.h"

#include "BVH/BVHNode.h"
#include "BVH/BVHTree.h"

#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"
#include "BVH/TopLevelBVH.h"

#include "Context.h"
