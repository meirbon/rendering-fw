cmake_minimum_required(VERSION 3.10)
project(glm)

file(GLOB_RECURSE SOURCES ./*.cpp ./*.h ./*.hpp)

add_library(${PROJECT_NAME} STATIC ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC ${PROJECT_SOURCE_DIR})