cmake_minimum_required(VERSION 3.10)
project(CUDART CXX CUDA)

find_package(CUDA REQUIRED)

if (WIN32)
	set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -use_fast_math -O3 -rdc=true --expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75)
else (WIN32)
    set(CUDA_NVCC_FLAGS_RELWITHDEBINFO ${CUDA_NVCC_FLAGS_RELWITHDEBINFO} -use_fast_math -O3 -rdc=true --expt-relaxed-constexpr -ccbin g++-8 -gencode arch=compute_75,code=sm_75)
endif (WIN32)

set(CUDA_NVCC_COMPILE_DEFINITIONS "${CUDA_NVCC_COMPILE_DEFINITIONS}" "GLM_FORCE_PURE")

file(GLOB_RECURSE SOURCES ./src/*.cpp ./src/*.cu)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp ./src/*.cuh)

add_library(${PROJECT_NAME} SHARED ${SOURCES} ${HEADERS})
if(${CMAKE_VERSION} VERSION_GREATER "3.16.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.16.0")
	target_precompile_headers(${PROJECT_NAME} PUBLIC src/PCH.h)
endif()

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)

target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT GLEW_STATIC)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src RenderSystem ../RenderContext ../RenderSystem/src ${CUDA_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

find_package(OpenGL REQUIRED)
if (WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW OpenGL::GL)
else ()
	find_package(GLEW REQUIRED)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW::GLEW OpenGL::GL)
endif ()

target_link_libraries(${PROJECT_NAME} PRIVATE ${CUDA_LIBRARIES})

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)

add_dependencies(Renderers ${PROJECT_NAME})