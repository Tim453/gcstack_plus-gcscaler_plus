
# trace-parser – protobuf-based dynamic trace reader (GCScaler variant,
#                no gpgpu_sim dependency).
# traces_enhanced and proto_objects are already defined by the parent build.

set(TRACES_ENHANCED_SRC_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../../gcstack/util/traces_enhanced/src)

add_library(trace-parser OBJECT
    trace-parser-enhanced/trace_parser.cc
)

target_include_directories(trace-parser
    PUBLIC  trace-parser-enhanced
    PRIVATE ${TRACES_ENHANCED_SRC_DIR}
)

target_link_libraries(trace-parser PUBLIC
    traces_enhanced
)

# ---------------------------------------------------------------------------
# trace-parser-legacy – text-based trace reader (old AccelSim format)
# ---------------------------------------------------------------------------
find_package(LibLZMA REQUIRED)
find_package(ZLIB REQUIRED)
find_package(BZip2 REQUIRED)

add_library(trace-parser-legacy OBJECT
    trace-parser-legacy/trace_parser_legacy.cc
)

target_include_directories(trace-parser-legacy
    PUBLIC  trace-parser-legacy
    PUBLIC  bxzstr
)

target_link_libraries(trace-parser-legacy PUBLIC
    LibLZMA::LibLZMA
    ZLIB::ZLIB
    BZip2::BZip2
)

# ---------------------------------------------------------------------------
# Header-only third-party libraries
# ---------------------------------------------------------------------------
add_library(dkm2 INTERFACE)
target_include_directories(dkm2 INTERFACE dkm)

add_library(argparse INTERFACE)
target_include_directories(argparse INTERFACE argparse)
