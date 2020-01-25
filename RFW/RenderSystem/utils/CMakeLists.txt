project(rfwUtils)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
find_package(OpenGL REQUIRED)

if (WIN32)
	set(LIBS ${LIBS} GLEW GLFW3 glm)
else ()
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	find_package(Threads REQUIRED)

	set(OpenGL_GL_PREFERENCE GLVND)
	find_package(glfw3 REQUIRED)
	set(LIBS ${LIBS} glm glfw Threads::Threads)
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif ()

target_compile_definitions(${PROJECT_NAME} PUBLIC GLEW_STATIC GLFW_DLL)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS} OpenGL::GL ${CMAKE_DL_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src ${LIBS})
