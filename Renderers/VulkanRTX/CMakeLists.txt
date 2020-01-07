cmake_minimum_required(VERSION 3.12)
project(VulkanRTX)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES ./src/*.cpp)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
target_precompile_headers(${PROJECT_NAME} PUBLIC src/Context.h)

if (WIN32)
	target_compile_definitions(${PROJECT_NAME} PRIVATE VK_USE_PLATFORM_WIN32_KHR)
endif (WIN32)
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT)

find_package(Vulkan REQUIRED)

if (WIN32)
	set(LIBS ${LIBS} GLEW GLFW3)
else ()
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)

	set(OpenGL_GL_PREFERENCE GLVND)
	find_package(glfw3 REQUIRED)
	find_package(GLEW REQUIRED)
	set(LIBS ${LIBS} GLEW::GLEW glfw Threads::Threads)
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_include_directories(${PROJECT_NAME} PUBLIC Vulkan::Vulkan ${Vulkan_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem Vulkan::Vulkan VulkanMemoryAllocator ${LIBS})

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)


find_program(GLSL_VALIDATOR glslangValidator REQUIRED)

# Find all shader source files
file(GLOB_RECURSE GLSL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/rt_*)
file(GLOB_RECURSE GLSL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*)

add_custom_command(
		OUTPUT "${ASSET_OUTPUT_DIR}/vkrtxshaders"
		PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory "${ASSET_OUTPUT_DIR}/vkrtxshaders"
)

# Setup shader compilation for all source files
foreach (GLSL ${GLSL_SOURCE_FILES})
	get_filename_component(FILE_NAME ${GLSL} NAME)
	set(SPIRV "${ASSET_OUTPUT_DIR}/vkrtxshaders/${FILE_NAME}.spv")
	add_custom_command(
			OUTPUT ${SPIRV}
			PRE_BUILD
			COMMAND ${GLSL_VALIDATOR} ${GLSL} -V110 -o ${SPIRV}
			DEPENDS ${GLSL_FILES}
	)
	list(APPEND SPIRV_BINARY_FILES ${SPIRV})
endforeach ()

# Add shaders as targets
add_custom_target(vkrtxShaders DEPENDS ${ASSET_OUTPUT_DIR}/vkrtxshaders ${SPIRV_BINARY_FILES})

# Make shader compilation a dependency of library
add_dependencies(${PROJECT_NAME} vkrtxShaders)
