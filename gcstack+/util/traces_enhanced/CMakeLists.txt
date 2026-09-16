
add_subdirectory(dynamic_trace)
set_target_properties(proto_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

file(GLOB SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cc)
add_library(traces_enhanced OBJECT
    ${SOURCES}
)
set_target_properties(traces_enhanced PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(traces_enhanced PRIVATE 
    ${CUDAToolkit_INCLUDE_DIRS}
    PUBLIC 
    src
)

target_link_libraries(traces_enhanced PUBLIC
    proto_objects
)