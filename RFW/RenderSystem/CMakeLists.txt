project(RenderSystem)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} OBJECT ${SOURCES} ${HEADERS})
if (${CMAKE_VERSION} VERSION_GREATER "3.16.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.16.0")
	target_precompile_headers(${PROJECT_NAME} PUBLIC src/rfw.h)
	target_precompile_headers(${PROJECT_NAME} PRIVATE src/Internal.h)
endif ()

add_subdirectory(utils)
add_subdirectory(math)
add_subdirectory(bvh)

add_subdirectory(RenderContext)

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
find_package(OpenGL REQUIRED)

if (UNIX)
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	set(OpenGL_GL_PREFERENCE GLVND)
endif (UNIX)

find_package(Threads REQUIRED)
find_package(FreeImage REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(assimp CONFIG REQUIRED)

find_path(TINY_GLTF_INCLUDE_DIR tiny_gltf.h)
find_path(STB_IMAGE_INCLUDE_DIR stb_image.h)

find_package(nlohmann_json CONFIG REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC nlohmann_json nlohmann_json::nlohmann_json)
target_include_directories(${PROJECT_NAME} PUBLIC nlohmann_json nlohmann_json::nlohmann_json)
target_include_directories(${PROJECT_NAME} PUBLIC ${TINY_GLTF_INCLUDE_DIR} ${STB_IMAGE_INCLUDE_DIR})
target_include_directories(${PROJECT_NAME} PRIVATE ${FREEIMAGE_INCLUDE_DIR})
if (UNIX)
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif (UNIX)

target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext rfwUtils rfwMath rfwUtils rfwBVH
		${FREEIMAGE_LIBRARIES} glfw Threads::Threads
		OpenGL::GL GLEW::GLEW ${CMAKE_DL_LIBS} Half ImGuiWrapper assimp::assimp)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src rtbvh)
target_include_directories(${PROJECT_NAME} PUBLIC assimp::assimp)

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)