project(OptiX_PTX)

SET(CUDA_VERBOSE_BUILD ON CACHE BOOL "nvcc verbose" FORCE)
SET(CMAKE_VERBOSE_MAKEFILE TRUE CACHE BOOL "verbose makefile" FORCE)

get_filename_component(ABSOLUTE_PATH "../../../../../Dependencies/OptiX6.5/include" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})
get_filename_component(ABSOLUTE_PATH "../../../../../Dependencies/half2.1.0" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})
get_filename_component(ABSOLUTE_PATH "../../src" ABSOLUTE)
include_directories(${ABSOLUTE_PATH})

set(ARCHES "52" "60" "61" "70" "75")
set(DEPS "")

foreach (ARCH ${ARCHES})
    set(OUTPUT_PTX "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels${ARCH}.ptx")
    cuda_compile_ptx(cuda_ptx ${PROJECT_SOURCE_DIR}/Kernels.cu OPTIONS -gencode=arch=compute_${ARCH},code=sm_${ARCH} -use_fast_math --expt-relaxed-constexpr)
    add_custom_command(
    		OUTPUT ${OUTPUT_PTX}
    		COMMAND ${CMAKE_COMMAND} -E copy ${cuda_ptx} ${OUTPUT_PTX}
    		DEPENDS ${cuda_ptx}
    )

    list(APPEND DEPS ${OUTPUT_PTX})
endforeach()

add_custom_target(OptiX_PTX ALL
		DEPENDS ${DEPS}
		SOURCES ./Kernels.cu
		)
