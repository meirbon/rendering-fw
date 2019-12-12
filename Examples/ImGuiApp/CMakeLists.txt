project(ImGuiApp)

add_executable(${PROJECT_NAME} main.cpp)

if (WIN32)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderSystem RenderContext GLFW3 GLEW ImGuiWrapper)
else (WIN32)
	find_package(glfw3 REQUIRED)
	find_package(GLEW REQUIRED)
	target_link_libraries(${PROJECT_NAME} PUBLIC RenderSystem RenderContext glfw GLEW::GLEW ImGuiWrapper)
endif (WIN32)
