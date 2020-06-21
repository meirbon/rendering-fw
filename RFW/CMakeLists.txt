add_subdirectory(RenderSystem)

add_custom_target(Renderers)

add_subdirectory(Renderers/CPURT)
find_library(EMBREE3 embree3)
if (EMBREE3)
	add_subdirectory(Renderers/EmbreeRT)
endif (EMBREE3)
add_subdirectory(Renderers/VkContext)
add_subdirectory(Renderers/VulkanRTX)
add_subdirectory(Renderers/GLRenderer)

if (ENABLE_CUDA)
	find_package(CUDA QUIET)
	if (${CUDA_FOUND})
		add_subdirectory(Renderers/OptiX6Context)
		add_subdirectory(Renderers/CUDART)
	endif (${CUDA_FOUND})
endif ()

add_dependencies(Renderers RenderSystem)