project(OptiX6.5)

add_library(OptiX6 SHARED IMPORTED GLOBAL)
add_library(OptiX6u SHARED IMPORTED GLOBAL)
add_library(OptiX6prime SHARED IMPORTED GLOBAL)

if (WIN32)
    set_target_properties(OptiX6 PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/windows/optix.6.5.0.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/windows/optix.6.5.0.lib
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )

    set_target_properties(OptiX6u PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/windows/optixu.6.5.0.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/windows/optixu.6.5.0.lib
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )

    set_target_properties(OptiX6prime PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/windows/optix_prime.6.5.0.dll
            IMPORTED_IMPLIB ${PROJECT_SOURCE_DIR}/windows/optix_prime.6.5.0.lib
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )
else ()
    set_target_properties(OptiX6 PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/linux/liboptix.so.6.5.0
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )

    set_target_properties(OptiX6u PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/linux/liboptixu.so.6.5.0
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )

    set_target_properties(OptiX6prime PROPERTIES
            IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/linux/liboptix_prime.so.6.5.0
            INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
            )
endif ()