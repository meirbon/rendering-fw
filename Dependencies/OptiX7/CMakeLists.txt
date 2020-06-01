project(OptiX7)
add_library(OptiX7 SHARED IMPORTED GLOBAL)

set_target_properties(OptiX7 PROPERTIES
		INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
)
