project(rfwBVH)

file(GLOB_RECURSE SOURCES src/*.cpp)
file(GLOB_RECURSE HEADERS src/*.h)

add_library(${PROJECT_NAME} STATIC ${SOURCES} ${HEADERS})

include(ExternalProject)
set_directory_properties(PROPERTIES EP_PREFIX ${CMAKE_BINARY_DIR}/Rust)

set(RTBVH_BINARY_PREFIX "${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/target")

if (APPLE)
	set(RTBVH_IMPL "${RTBVH_BINARY_PREFIX}/release/librtbvh_rs.dylib")
	list(APPEND RTBVH_BYPRODUCTS ${RTBVH_IMPL})
elseif (UNIX)
	set(RTBVH_IMPL "${RTBVH_BINARY_PREFIX}/release/librtbvh_rs.so")
	list(APPEND RTBVH_BYPRODUCTS ${RTBVH_IMPL})
elseif (WIN32)
	set(RTBVH_IMPL "${RTBVH_BINARY_PREFIX}/release/rtbvh_rs.dll")
	set(RTBVH_LIB "${RTBVH_BINARY_PREFIX}/release/rtbvh_rs.dll.lib")
	list(APPEND RTBVH_BYPRODUCTS ${RTBVH_IMPL})
	list(APPEND RTBVH_BYPRODUCTS ${RTBVH_LIB})
endif ()

ExternalProject_Add(
		rtbvh_rs
		GIT_REPOSITORY "https://github.com/MeirBon/rtbvh.git"
		GIT_TAG "0.5.2"
		GIT_SHALLOW ON
		CONFIGURE_COMMAND ""
		BUILD_COMMAND cargo build --quiet --manifest-path=${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/Cargo.toml
		COMMAND cargo build --release --quiet --manifest-path=${CMAKE_BINARY_DIR}/Rust/src/rtbvh_rs/rtbvh_ffi/Cargo.toml
		INSTALL_COMMAND ""
		LOG_BUILD ON
		BUILD_BYPRODUCTS ${RTBVH_BYPRODUCTS}
		UPDATE_DISCONNECTED ON
)

add_library(rtbvh SHARED IMPORTED GLOBAL)

add_dependencies(rtbvh rtbvh_rs)

set_property(TARGET rtbvh PROPERTY IMPORTED_LOCATION ${RTBVH_IMPL})
if (WIN32)
	set_property(TARGET rtbvh PROPERTY IMPORTED_IMPLIB ${RTBVH_LIB})
endif (WIN32)

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