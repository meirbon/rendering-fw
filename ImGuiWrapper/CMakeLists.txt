project(ImGuiWrapper)

set(SOURCES 
	src/imconfig.h
	src/imgui.cpp
	src/imgui.h
	src/imgui_demo.cpp
	src/imgui_draw.cpp
	src/imgui_internal.h
	src/imgui_widgets.cpp
	src/imstb_rectpack.h
	src/imstb_textedit.h
	src/imstb_truetype.h
	src/imgui_impl_glfw.cpp
	src/imgui_impl_glfw.h
	src/imgui_impl_opengl3.cpp
	src/imgui_impl_opengl3.h
	src/imgui_impl_vulkan.cpp
	src/imgui_impl_vulkan.h
	wrapper/ImGuiContext.h
	)

if (APPLE)
	set(SOURCES ${SOURCES} wrapper/ImGuiContext.mm src/imgui_impl_metal.h src/imgui_impl_metal.mm)
else(APPLE)
	set(SOURCES ${SOURCES} wrapper/ImGuiContext.cpp)
endif(APPLE)

add_library(${PROJECT_NAME} STATIC ${SOURCES})

find_package(Vulkan REQUIRED)

if (WIN32)
	set(LIBS GLFW3 Vulkan::Vulkan GLEW)
else(WIN32)
	find_package(glfw3 REQUIRED)
	find_package(GLEW REQUIRED)
	set(LIBS glfw Vulkan::Vulkan GLEW::GLEW)
	target_compile_options(${PROJECT_NAME} PRIVATE "-fPIC")
endif(WIN32)

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