cmake_minimum_required(VERSION 3.10)
if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
            CACHE STRING "")
endif ()

project(RenderingFW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH}" "${CMAKE_CURRENT_SOURCE_DIR}/CMake")
option(ENABLE_PROFILING "Enable profiling support for Visual Studio" OFF)

find_package(CUDA)
if (${CUDA_FOUND})
    option(ENABLE_CUDA "Enable compilation of CUDA-based renderers" ON)
else (${CUDA_FOUND})
    option(ENABLE_CUDA "Enable compilation of CUDA-based renderers" OFF)
endif (${CUDA_FOUND})

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang") # using Clang
    if (WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast -mavx2 /Oi /Ot /GL /openmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast -mavx2 /Oi /Ot /GL /openmp")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} /fp:fast -mavx2 /Oi /Ot /GL /openmp")

        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /NODEFAULTLIB /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /LTCG")

        if (ENABLE_PROFILING)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
            set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /PROFILE")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /PROFILE")
            set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /PROFILE")
        endif (ENABLE_PROFILING)
    elseif (APPLE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -mavx2 -march=core-avx2")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -mavx2 -march=core-avx2")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -ffast-math -mavx2 -march=core-avx2")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -mavx2 -march=core-avx2 -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -mavx2 -march=core-avx2 -fopenmp")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -ffast-math -mavx2 -march=core-avx2 -fopenmp")
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU") # using gcc
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffast-math -mavx2 -fopenmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math -mavx2 -fopenmp")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -ffast-math -mavx2 -fopenmp")

    if (ENABLE_PROFILING)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -pg")
    endif (ENABLE_PROFILING)
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel") # using Intel
    if (WIN32)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast=1 -march=core-avx2 /Oi /Ot /GL /openmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast=1 -march=core-avx2 /Oi /Ot /GL /openmp")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} /fp:fast=1 -march=core-avx2 /Oi /Ot /GL /openmp")

        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /NODEFAULTLIB /LTCG")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /LTCG")

        if (ENABLE_PROFILING)
            set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
            set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /PROFILE")
            set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /PROFILE")
            set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /PROFILE")
        endif (ENABLE_PROFILING)
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fp-model:fast=1 -march=core-avx2 -fopenmp")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fp-model:fast=1 -march=core-avx2 -fopenmp")
        set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -fp-model:fast=1 -march=core-avx2 -fopenmp")

        if (ENABLE_PROFILING)
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -pg")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
            set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} -pg")
        endif (ENABLE_PROFILING)
    endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "MSVC") # using MSVC
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /fp:fast /arch:AVX2 /Oi /Ot /GL /openmp")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /fp:fast /arch:AVX2 /Oi /Ot /GL /openmp")
    set(CMAKE_ASM_FLAGS "${CMAKE_ASM_FLAGS} /fp:fast /arch:AVX2 /Oi /Ot /GL /openmp")

    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
    set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /NODEFAULTLIB /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /LTCG ")
    set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /LTCG")

    if (ENABLE_PROFILING)
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /PROFILE")
        set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /PROFILE")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /PROFILE")
        set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /PROFILE")
    endif (ENABLE_PROFILING)
endif ()

if (WIN32)
    add_definitions("-D_CRT_SECURE_NO_WARNINGS" "-DWIN32")
else (WIN32)
    set(OpenGL_GL_PREFERENCE GLVND)
endif (WIN32)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(ASSET_OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(Dependencies)
add_subdirectory(RFW)

add_subdirectory(Examples/App)
add_subdirectory(Examples/ImGuiApp)
#if (APPLE)
#	add_subdirectory(Examples/MetalApp)
#endif (APPLE)

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

add_dependencies(App Renderers)
add_dependencies(ImGuiApp Renderers)