project(OptiX_PTX)

SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "verbose makefile" FORCE)

get_filename_component(ABSOLUTE_PATH "../../../../Dependencies/OptiX6.5/include" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})
get_filename_component(ABSOLUTE_PATH "../../../../Dependencies/half2.1.0" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})
get_filename_component(ABSOLUTE_PATH "../../src" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})

cuda_compile_ptx(cuda_ptx_files ${PROJECT_SOURCE_DIR}/Kernels.cu OPTIONS -lineinfo -gencode arch=compute_75,code=sm_75 -use_fast_math --expt-relaxed-constexpr)
add_custom_command(
		OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		COMMAND ${CMAKE_COMMAND} -E copy ${cuda_ptx_files} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		DEPENDS ${cuda_ptx_files}
)

add_custom_target(OptiX_PTX ALL
		DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		SOURCES ./Kernels.cu
		)
