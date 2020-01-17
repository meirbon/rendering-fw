cmake_minimum_required(VERSION 3.12)
project(MetalRast)
set(CMAKE_CXX_STANDARD 17)

file(GLOB_RECURSE SOURCES ./src/*.cpp ./src/*.mm)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT)

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)

find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)
find_library(APPKIT AppKit)
find_library(METAL Metal)
find_library(METALKIT MetalKit)
find_library(IOKIT IOKit)
find_library(COCOA Cocoa)

set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

target_link_libraries(${PROJECT_NAME} PRIVATE ${LIBS})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem glfw Threads::Threads GLEW::GLEW)
target_link_libraries(${PROJECT_NAME} PRIVATE
		${FOUNDATION} ${QUARTZCORE} ${APPKIT} ${METAL} ${METALKIT} ${IOKIT} ${COREVIDEO} ${COCOA}
		)

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)

add_custom_command(
		OUTPUT "${ASSET_OUTPUT_DIR}/metalshaders"
		PRE_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory "${ASSET_OUTPUT_DIR}/vkshaders"
)

file(GLOB SHADERS ${PROJECT_SOURCE_DIR}/shaders/*)

foreach (shader ${SHADERS})
	get_filename_component(basename ${shader} NAME)
	configure_file(${shader} ${ASSET_OUTPUT_DIR}/metalshaders/${basename})
endforeach ()

add_dependencies(Renderers ${PROJECT_NAME})