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

#include <RenderContext.h>
#include <ContextExport.h>

#include <utils/gl/CheckGL.h>
#include <MathIncludes.h>

#include <tbb/tbb.h>

#include <utils.h>

#include "Ray.h"

#include <bvh/AABB.h>

#include "Triangle.h"

#include <bvh/BVHNode.h>
#include <bvh/BVHTree.h>

#include <bvh/MBVHNode.h>
#include <bvh/MBVHTree.h>
#include <bvh/TopLevelBVH.h>

#include "Context.h"