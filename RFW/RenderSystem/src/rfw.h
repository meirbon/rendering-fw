#pragma once

// Pre-compiled header file for RFW RenderSystem

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

#include <Structures.h>
#include <RenderContext.h>
#include <ContextExport.h>

#include "GeometryReference.h"
#include "InstanceReference.h"
#include "LightReference.h"

#include "utils/gl/CheckGL.h"
#include "utils/gl/GLBuffer.h"
#include "utils/gl/GLDraw.h"
#include "utils/gl/GLShader.h"
#include "utils/gl/GLTexture.h"
#include "utils/gl/GLTextureArray.h"
#include "utils/gl/VertexArray.h"

#include "utils/ArrayProxy.h"
#include "utils/Averager.h"
#include "utils/Concurrency.h"
#include "utils/File.h"
#include "utils/gl.h"
#include "utils/LibExport.h"
#include "utils/Logger.h"
#include "utils/MersenneTwister.h"
#include "utils/RandomGenerator.h"
#include "utils/Serializable.h"
#include "utils/String.h"
#include "utils/ThreadPool.h"
#include "utils/Time.h"
#include "utils/Timer.h"
#include "utils/Window.h"
#include "utils/Xor128.h"

#include "MaterialList.h"
#include "MathIncludes.h"
#include "Quad.h"
#include "SceneTriangles.h"
#include "Settings.h"
#include "Skybox.h"
#include "Texture.h"
#include "../utils/src/utils.h"

#include "Application.h"
#include "AssimpObject.h"
#include "BlueNoise.h"

#include "gLTF/gLTFObject.h"
#include "gLTF/SceneAnimation.h"
#include "gLTF/SceneMesh.h"
#include "gLTF/SceneNode.h"
#include "gLTF/SceneObject.h"
#include "gLTF/Skinning.h"

#include "RenderSystem.h"

#include "bvh/BVH.h"