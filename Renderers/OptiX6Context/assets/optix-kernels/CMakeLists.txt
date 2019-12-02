project(OptiX_PTX)

include_directories(../../../Dependencies/OptiX6.5/include ../../../Dependencies/glm-9.9.6 ../../../RenderContext ../../../RenderSystem/src ../../../Dependencies/half2.1.0 ../../src)

cuda_compile_ptx(cuda_ptx_files ${PROJECT_SOURCE_DIR}/Kernels.cu OPTIONS -lineinfo -gencode arch=compute_75,code=sm_75 -use_fast_math --expt-relaxed-constexpr)
add_custom_command(
		OUTPUT ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		COMMAND ${CMAKE_COMMAND} -E copy ${cuda_ptx_files} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		DEPENDS ${cuda_ptx_files}
)

add_custom_target(ptx ALL
		DEPENDS ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Kernels.ptx
		SOURCES ./Kernels.cu
		)
