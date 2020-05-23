project(rfwUtils)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})


set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Threads REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(TBB CONFIG REQUIRED)

if (UNIX)
	set(THREADS_PREFER_PTHREAD_FLAG ON)
	set(OpenGL_GL_PREFERENCE GLVND)
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif (UNIX)

target_link_libraries(${PROJECT_NAME} PUBLIC OpenGL::GL ${CMAKE_DL_LIBS} GLEW::GLEW glfw Threads::Threads TBB::tbb)
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/src OpenGL::GL ${CMAKE_DL_LIBS} GLEW::GLEW glm glfw Threads::Threads glm TBB::tbb)
