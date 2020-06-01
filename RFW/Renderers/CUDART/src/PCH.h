#pragma once

// Include files in this header
// Only include this header in every file for faster compilation

#include <utils.h>

#include <optional>
#include <utils/Window.h>
#include <utils/LibExport.h>
#include <utils/gl/GLDraw.h>
#include <utils/gl/GLTexture.h>

#include <RenderContext.h>
#include <ContextExport.h>

#include <bvh/BVH.h>
#include <BlueNoise.h>

#include <GL/glew.h>

#include "CheckCUDA.h"

#include "CUDABuffer.h"

#include "Shared.h"

#include "Ray.h"

#include "TextureInterop.h"
#include "Context.h"