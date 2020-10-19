project(ImGuiApp)

add_executable(${PROJECT_NAME} main.cpp)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC RenderSystem RenderContext glfw GLEW::GLEW ImGuiWrapper)
target_include_directories(${PROJECT_NAME} PUBLIC RenderSystem RenderContext glfw GLEW::GLEW ImGuiWrapper)