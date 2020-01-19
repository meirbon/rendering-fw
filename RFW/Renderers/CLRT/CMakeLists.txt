cmake_minimum_required(VERSION 3.10)
project(CLRT)

file(GLOB_RECURSE SOURCES ./src/*.cpp)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS} clkernels/shared.h)
if (${CMAKE_VERSION} VERSION_GREATER "3.16.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.16.0")
	target_precompile_headers(${PROJECT_NAME} PUBLIC src/PCH.h)
endif ()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT GLEW_STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src RenderSystem ../RenderContext ../RenderSystem/src)

find_package(OpenGL REQUIRED)
if (WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW OpenGL::GL)
else ()
	find_package(GLEW REQUIRED)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW::GLEW OpenGL::GL)
endif ()

find_package(OpenCL REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenCL::OpenCL)

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)

add_custom_target(LINK_CL_KERNELS
		COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/clkernels ${ASSET_OUTPUT_DIR}/clkernels
		)

add_dependencies(${PROJECT_NAME} LINK_CL_KERNELS)
add_dependencies(Renderers ${PROJECT_NAME})