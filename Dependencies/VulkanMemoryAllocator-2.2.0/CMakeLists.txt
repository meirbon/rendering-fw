cmake_minimum_required(VERSION 3.10)
project(VulkanMemoryAllocator)

file(GLOB_RECURSE SOURCES ./src/*.cpp ./src/*.h ./src/*.hpp)

find_package(Vulkan REQUIRED)

add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_link_libraries(${PROJECT_NAME} PUBLIC Vulkan::Vulkan)
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include Vulkan::Vulkan)