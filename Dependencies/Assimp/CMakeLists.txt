project(assimp410)
add_library(assimp SHARED IMPORTED GLOBAL)

set_target_properties(assimp PROPERTIES
		IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/assimp-vc140-mt.dll
		IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/assimp-vc140-mt.lib
		INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
		)