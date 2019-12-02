cmake_minimum_required(VERSION 3.12)
project(OptiX6Context CXX CUDA)

find_package(CUDA REQUIRED)

if (WIN32)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math -rdc=true --expt-relaxed-constexpr)
else (WIN32)
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -use_fast_math -rdc=true --expt-relaxed-constexpr -ccbin g++-8 -gencode arch=compute_75,code=sm_75)
endif (WIN32)

file(GLOB_RECURSE SOURCES ./src/*.cpp)
file(GLOB_RECURSE KERNELS ./assets/kernels/*.cu ./assets/kernels/*.h)
file(GLOB_RECURSE HEADERS ./src/*.h ./src/*.hpp)
add_library(${PROJECT_NAME} SHARED ${SOURCES} ${KERNELS} ${HEADERS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
set_target_properties(${PROJECT_NAME} PROPERTIES
		POSITION_INDEPENDENT_CODE ON
		CUDA_SEPARABLE_COMPILATION ON)
target_compile_definitions(${PROJECT_NAME} PRIVATE RENDER_API_EXPORT)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDA_TOOLKIT_ROOT_DIR}/include RenderSystem ../RenderContext ../RenderSystem/src)
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/assets/kernels)
if (WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW OptiX6 OptiX6u OptiX6prime ${CUDA_LIBRARIES})
else ()
	find_package(GLEW REQUIRED)
	set(OptiX_INSTALL_DIR "/opt/optix65" CACHE PATH "Path to OptiX installed location.")
	set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} /opt/optix65/SDK/CMake /opt/optix6/SDK/CMake)
	find_package(OptiX REQUIRED)
	target_include_directories(${PROJECT_NAME} PRIVATE ${OptiX_INSTALL_DIR}/include)
	target_link_libraries(${PROJECT_NAME} PRIVATE optix_prime)
	target_link_libraries(${PROJECT_NAME} PRIVATE optix)

	target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext RenderSystem GLEW::GLEW ${CUDA_LIBRARIES})
endif ()


target_compile_options(${PROJECT_NAME} PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-expt-relaxed-constexpr>)
set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)

add_subdirectory(assets/optix-kernels)