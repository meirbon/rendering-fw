project(VMA)

add_library(${PROJECT_NAME} vma.cpp)

find_package(Vulkan REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC ./)
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)