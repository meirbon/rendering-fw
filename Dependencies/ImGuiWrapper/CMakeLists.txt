project(ImGuiWrapper)

if (APPLE)
	set(SOURCES wrapper/ImGuiContext.mm wrapper/imgui_impl_metal.h wrapper/imgui_impl_metal.mm)
else(APPLE)
	set(SOURCES wrapper/ImGuiContext.cpp
		wrapper/imgui_impl_glfw.cpp
		wrapper/imgui_impl_opengl3.cpp
		wrapper/imgui_impl_vulkan.cpp
		)
endif(APPLE)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

find_package(imgui CONFIG REQUIRED)
find_package(Vulkan REQUIRED)
find_package(glfw3 REQUIRED)
find_package(GLEW REQUIRED)
set(LIBS imgui::imgui glfw Vulkan::Vulkan GLEW::GLEW)

if (UNIX)
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif (UNIX)

if(APPLE)
	find_library(FOUNDATION Foundation)
	find_library(QUARTZCORE QuartzCore)
	find_library(APPKIT AppKit)
	find_library(METAL Metal)
	find_library(METALKIT MetalKit)
	find_library(IOKIT IOKit)
	find_library(COCOA Cocoa)
	set(LIBS ${LIBS} ${FOUNDATION} ${QUARTZCORE} ${APPKIT} ${METAL} ${METALKIT} ${IOKIT} ${COREVIDEO} ${COCOA})
endif(APPLE)

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_include_directories(${PROJECT_NAME} PUBLIC wrapper)
target_link_libraries(${PROJECT_NAME} PUBLIC ${LIBS})