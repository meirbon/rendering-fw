project(RenderSystem)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
find_package(OpenGL REQUIRED)

if (WIN32)
	set(LIBS ${LIBS} GLEW assimp FreeImage GLFW3 glm)
else ()
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)

	set(OpenGL_GL_PREFERENCE GLVND)
	find_package(FreeImage REQUIRED)
	find_package(ASSIMP REQUIRED)
	find_package(glfw3 REQUIRED)
	set(LIBS ${LIBS} glm ${FREEIMAGE_LIBRARIES} ${ASSIMP_LIBRARIES} glfw Threads::Threads)
	target_include_directories(${PROJECT_NAME} PRIVATE ${ASSIMP_INCLUDE_DIR} ${FREEIMAGE_INCLUDE_DIR})
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif ()

target_link_libraries(${PROJECT_NAME} PUBLIC RenderContext ${LIBS} OpenGL::GL ${CMAKE_DL_LIBS} Half tinygltf)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src ${LIBS} tinygltf)

set_target_properties(${PROJECT_NAME}
		PROPERTIES
		PREFIX ""
		ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		LIBRARY_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		RUNTIME_OUTPUT_DIRECTORY ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
		CXX_STANDARD 17
		)