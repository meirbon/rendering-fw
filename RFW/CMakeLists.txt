add_subdirectory(system)

add_custom_target(backends)

add_subdirectory(backends/CPURT)
add_subdirectory(backends/EmbreeRT)
if (NOT APPLE)
	add_subdirectory(backends/VkContext)
	add_subdirectory(backends/VulkanRTX)
endif (NOT APPLE)
add_subdirectory(backends/GLRenderer)

if (ENABLE_CUDA)
	find_package(CUDA QUIET)
	if (${CUDA_FOUND})
		add_subdirectory(backends/OptiX6Context)
		add_subdirectory(backends/CUDART)
	endif (${CUDA_FOUND})
endif ()