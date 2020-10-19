#pragma once

// Pre-compiled header file for RFW system

#ifdef _WIN32
#include <Windows.h>
#endif

#define GLFW_INCLUDE_VULKAN 1

#include <utility>
#include <vector>
#include <string>
#include <string_view>
#include <bitset>
#include <mutex>
#include <future>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#include <tbb/tbb.h>

#include <rfw/context/structs.h>
#include <rfw/context/context.h>
#include <rfw/context/export.h>

#include "geometry_ref.h"
#include "instance_ref.h"
#include "light_ref.h"

#include <rfw/utils/gl/check.h>
#include <rfw/utils/gl/buffer.h>
#include <rfw/utils/gl/draw.h>
#include <rfw/utils/gl/shader.h>
#include <rfw/utils/gl/texture.h>
#include <rfw/utils/gl/texture_array.h>
#include <rfw/utils/gl/vertex_array.h>

#include <rfw/utils/array_proxy.h>
#include <rfw/utils/averager.h>
#include <rfw/utils/file.h>
#include <rfw/utils/gl.h>
#include <rfw/utils/lib_export.h>
#include <rfw/utils/logger.h>
#include <rfw/utils/mersenne_twister.h>
#include <rfw/utils/rng.h>
#include <rfw/utils/serializable.h>
#include <rfw/utils/string.h>
#include <rfw/utils/thread_pool.h>
#include <rfw/utils/time.h>
#include <rfw/utils/timer.h>
#include <rfw/utils/window.h>
#include <rfw/utils/xor128.h>

#include "material_list.h"
#include <rfw_math.h>
#include "geometry/quad.h"
#include "geometry/triangles.h"
#include <rfw/context/settings.h>
#include "skybox.h"
#include "texture.h"

#include "rfw/app.h"
#include <rfw/context/blue_noise.h>

#include "geometry/assimp/object.h"
#include "geometry/gltf/object.h"
#include "geometry/gltf/animation.h"
#include "geometry/gltf/mesh.h"
#include "geometry/gltf/node.h"
#include "geometry/gltf/hierarcy.h"
#include "geometry/gltf/skinning.h"

#include "system.h"

#include "bvh/BVH.h"