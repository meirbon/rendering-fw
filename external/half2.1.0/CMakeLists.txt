project(Half)
add_library(${PROJECT_NAME} INTERFACE IMPORTED GLOBAL)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
