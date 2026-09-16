
set(CXX_STANDARD 20)

add_executable(gcscaler_plus
    cache_model.cc
    common.cc
    inst_decoder.cc
    instruction.cc
    interval_model.cc
    main.cc
    worker.cc
)

find_package(Boost REQUIRED COMPONENTS iostreams serialization)

target_include_directories(gcscaler_plus PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include 
    )

target_link_libraries(gcscaler_plus PUBLIC
    trace-parser
    trace-parser-legacy
    Boost::serialization
    dkm2
    argparse
    traces_enhanced
)

target_compile_options(gcscaler_plus PRIVATE -Wall -Wextra -Wpedantic)

if(NOT CMAKE_BUILD_TYPE MATCHES "Debug")
    set_property(TARGET gcscaler_plus PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    target_compile_options(gcscaler_plus PRIVATE "-march=native")
endif()


set_target_properties(gcscaler_plus PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)