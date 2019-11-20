cmake_minimum_required(VERSION 3.10)
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
	set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
			CACHE STRING "")
endif ()

project(RenderingFW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
if (WIN32)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)

	set(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
	set(CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS} /PROFILE)
	set(CMAKE_MODULE_LINKER_FLAGS ${CMAKE_MODULE_LINKER_FLAGS} /PROFILE)
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
add_subdirectory(VkContext)
add_subdirectory(VulkanRTX)

if (APPLE)
	add_subdirectory(Examples/MetalApp)
else (APPLE)
	add_subdirectory(Examples/SimpleApp)
	add_dependencies(SimpleApp VulkanRTX)

	add_subdirectory(Examples/VulkanApp)
endif (APPLE)

if (${CUDA_FOUND})
	add_subdirectory(OptiX6Context)
	add_dependencies(SimpleApp OptiX6Context)
endif (${CUDA_FOUND})
if (APPLE)
	add_subdirectory(MetalRast)
	add_dependencies(SimpleApp MetalRast)
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