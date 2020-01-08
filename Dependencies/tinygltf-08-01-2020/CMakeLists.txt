project(tinygltf-08-01-2020)

file(GLOB_RECURSE SOURCES ./*.cpp ./*.h ./*.hpp)

add_library(tinygltf INTERFACE)
target_include_directories(tinygltf INTERFACE ${PROJECT_SOURCE_DIR})