cmake_minimum_required(VERSION 3.12)
project(GLRenderer)

file(GLOB_RECURSE SOURCES ./src/*.cpp)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src RenderSystem ../RenderContext ../RenderSystem/src)

find_package(OpenGL REQUIRED)
if (WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW OpenGL::GL)
else ()
	find_package(GLEW REQUIRED)

	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW::GLEW OpenGL::GL)
endif ()

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)

add_custom_command(
		TARGET ${PROJECT_NAME}
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/shaders ${ASSET_OUTPUT_DIR}/glshaders
)

## Find all shader source files
#file(GLOB_RECURSE GLSL_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/shaders/*)
#
#add_custom_target(glrShaders)
#
#
## Setup shader compilation for all source files
#foreach (GLSL ${GLSL_SOURCE_FILES})
#	get_filename_component(FILE_NAME ${GLSL} NAME)
#	set(OUTPUT " ${ASSET_OUTPUT_DIR}/glshaders/${FILE_NAME}")
#	message(${GLSL} ${OUTPUT})
#	add_custom_command(
#			TARGET glrShaders
#			POST_BUILD
#			COMMAND ${CMAKE_COMMAND} -E create_symlink ${GLSL} ${OUTPUT}
#			DEPENDS ${OUTPUT}
#	)
#	configure_file(${GLSL} ${OUTPUT})
#	list(APPEND SHADER_FILES ${OUTPUT})
#endforeach ()
#
## Make shader compilation a dependency of library
#add_dependencies(${PROJECT_NAME} glrShaders)
