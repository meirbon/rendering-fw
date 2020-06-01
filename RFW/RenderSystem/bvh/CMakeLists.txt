project(rfwBVH)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})
if (${CMAKE_VERSION} VERSION_GREATER "3.16.0" OR ${CMAKE_VERSION} VERSION_EQUAL "3.16.0")
	target_precompile_headers(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include/bvh/BVH.h")
endif ()

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/Rust)

set(RTBVH_BINARY_PREFIX "${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/target")

ExternalProject_Add(
		rtbvh_rs
		GIT_REPOSITORY "https://github.com/MeirBon/rtbvh.git"
		GIT_TAG "master"
		GIT_SHALLOW ON
		CONFIGURE_COMMAND ""
		BUILD_COMMAND cargo build --manifest-path=${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/Cargo.toml
		COMMAND cargo build --release --manifest-path=${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/Cargo.toml
		INSTALL_COMMAND ""
		LOG_BUILD ON
		BUILD_BYPRODUCTS ${RTBVH_BINARY_PREFIX}/release/rtbvh.dll ${RTBVH_BINARY_PREFIX}/release/rtbvh.dll.lib
)

add_dependencies(${PROJECT_NAME} rtbvh_rs)

add_library(rtbvh SHARED IMPORTED GLOBAL)
set_property(TARGET rtbvh PROPERTY IMPORTED_LOCATION ${RTBVH_BINARY_PREFIX}/release/rtbvh.dll)
set_property(TARGET rtbvh PROPERTY IMPORTED_IMPLIB ${RTBVH_BINARY_PREFIX}/release/rtbvh.dll.lib)

add_custom_command(TARGET rtbvh_rs
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${RTBVH_BINARY_PREFIX}/rtbvh.h" "${PROJECT_SOURCE_DIR}/include/rtbvh.h"
		)
add_custom_command(TARGET rtbvh_rs
		POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${RTBVH_BINARY_PREFIX}/rtbvh.hpp" "${PROJECT_SOURCE_DIR}/include/rtbvh.hpp"
		)

find_package(glm CONFIG REQUIRED)


target_link_libraries(${PROJECT_NAME} PUBLIC rtbvh)
target_include_directories(${PROJECT_NAME} PUBLIC rtbvh)

target_link_libraries(${PROJECT_NAME} PUBLIC glm RenderContext rfwMath rfwUtils)
target_include_directories(${PROJECT_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include" glm RenderContext rfwMath)