cmake_minimum_required(VERSION 3.28)

project(AccelSim VERSION 12.0 LANGUAGES CXX C CUDA)




set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
#set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


add_subdirectory(gpgpu-sim)

add_executable(gcstack_plus
    main.cc
    trace-driven/trace_driven.cc
    trace-parser/trace_parser.cc
)
target_link_libraries(gcstack_plus PUBLIC 
    cudart
    OpenMP::OpenMP_CXX
    OpenMP::OpenMP_C
    Threads::Threads
    traces_enhanced
    Boost::iostreams
    Boost::serialization
)
target_include_directories(gcstack_plus PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu-sim/src
    ${CMAKE_CURRENT_SOURCE_DIR}/gpgpu-sim/libcuda
    ${CMAKE_CURRENT_SOURCE_DIR}/ISA_Def
    ${CMAKE_CURRENT_SOURCE_DIR}/trace-driven
    ${CUDAToolkit_INCLUDE_DIRS}
)

set_target_properties(gcstack_plus PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)