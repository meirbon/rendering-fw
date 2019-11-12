project(MetalApp)

add_executable(${PROJECT_NAME} main.mm)

find_library(FOUNDATION Foundation)
find_library(QUARTZCORE QuartzCore)
find_library(APPKIT AppKit)
find_library(METAL Metal)
find_library(METALKIT MetalKit)
find_library(IOKIT IOKit)
find_library(COCOA Cocoa)

target_link_libraries(${PROJECT_NAME} PRIVATE ${FOUNDATION}
		${QUARTZCORE} ${APPKIT} ${METAL} ${METALKIT} ${IOKIT} ${COREVIDEO} ${COCOA})

find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
target_link_libraries(${PROJECT_NAME} PUBLIC RenderSystem RenderContext glfw GLEW::GLEW ImGuiWrapper)
