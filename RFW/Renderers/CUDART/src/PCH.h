#pragma once

// Include files in this header
// Only include this header in every file for faster compilation

#include <rfw.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>
#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>

#include <GL/glew.h>

#include "CheckCUDA.h"

#include "CUDABuffer.h"

#include "Shared.h"

#include "Ray.h"

#include "BVH/AABB.h"

#include "Triangle.h"

#include "BVH/BVHNode.h"
#include "BVH/BVHTree.h"

#include "BVH/MBVHNode.h"
#include "BVH/MBVHTree.h"
#include "BVH/TopLevelBVH.h"

#include "TextureInterop.h"

#include "Context.h"