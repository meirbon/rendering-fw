add_subdirectory(RenderContext)
add_subdirectory(RenderSystem)

add_custom_target(Renderers)

add_subdirectory(Renderers/CPURT)
add_subdirectory(Renderers/EmbreeRT)
add_subdirectory(Renderers/VkContext)
add_subdirectory(Renderers/VulkanRTX)
add_subdirectory(Renderers/GLRenderer)

find_package(CUDA QUIET)
if (${CUDA_FOUND})
	add_subdirectory(Renderers/OptiX6Context)
	add_subdirectory(Renderers/CUDART)
endif (${CUDA_FOUND})

if (APPLE)
	add_subdirectory(Renderers/MetalRast)
endif (APPLE)
