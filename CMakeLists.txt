cmake_minimum_required(VERSION 3.10)
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
			CACHE STRING "")
endif ()

project(RenderingFW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/CMake")

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang") # using Clang
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -msse4.2 -mavx2 -march=core-avx2")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2 -mavx2 -march=core-avx2")
	set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -msse4.2 -mavx2 -march=core-avx2")
endif ()

if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
else (WIN32)
	set(OpenGL_GL_PREFERENCE GLVND)
endif (WIN32)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(ASSET_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(ImGuiWrapper)

find_package(CUDA)
add_subdirectory(Dependencies)
add_subdirectory(RenderContext)
add_subdirectory(RenderSystem)
add_subdirectory(Renderers/CPURT)
add_subdirectory(Renderers/VkContext)
add_subdirectory(Renderers/VulkanRTX)
add_subdirectory(Renderers/GLRenderer)

add_subdirectory(Examples/App)
add_subdirectory(Examples/ImGuiApp)

add_dependencies(App CPURT)
add_dependencies(App VkContext)
add_dependencies(App VulkanRTX)
add_dependencies(App GLRenderer)

add_dependencies(ImGuiApp CPURT)
add_dependencies(ImGuiApp VkContext)
add_dependencies(ImGuiApp VulkanRTX)
add_dependencies(ImGuiApp GLRenderer)

#if (APPLE)
#	add_subdirectory(Examples/MetalApp)
#else (APPLE)
#	add_subdirectory(Examples/VulkanApp)
#endif (APPLE)

if (${CUDA_FOUND})
	add_subdirectory(Renderers/OptiX6Context)
	add_dependencies(App OptiX6Context)
	add_dependencies(ImGuiApp OptiX6Context)
endif (${CUDA_FOUND})
if (APPLE)
	add_subdirectory(Renderers/MetalRast)
endif (APPLE)

add_custom_target(CopyCommonAssets)

file(GLOB ASSETS ${PROJECT_SOURCE_DIR}/CommonAssets/*)
foreach (asset ${ASSETS})
	get_filename_component(FOLDER_NAME ${asset} NAME)
	add_custom_command(
			TARGET CopyCommonAssets
			POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E create_symlink ${asset} ${ASSET_OUTPUT_DIR}/${FOLDER_NAME}
			DEPENDS ${asset})
endforeach ()

add_dependencies(RenderContext CopyCommonAssets)