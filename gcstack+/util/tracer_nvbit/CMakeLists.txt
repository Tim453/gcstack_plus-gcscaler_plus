cmake_minimum_required(VERSION 3.19)

project(tracer_nvbit VERSION 12.0 LANGUAGES CXX C CUDA)
find_package(CUDAToolkit REQUIRED)
include(FetchContent)

FetchContent_Declare(
    nvbit
    URL "https://github.com/NVlabs/NVBit/releases/download/v1.7.7.3/nvbit-Linux-x86_64-1.7.7.3.tar.bz2"
    # Optional: Use hash verification for security/integrity
    # URL_HASH "SHA256=..." 
)
FetchContent_MakeAvailable(nvbit)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "all")



add_executable(enhanced_post-traces-processing tracer_tool/traces-processing/post-traces-processing.cpp)
set_target_properties(enhanced_post-traces-processing PROPERTIES 
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
target_link_libraries(enhanced_post-traces-processing PUBLIC traces_enhanced)


# Check PTXAS version for register limit flag (Legacy support)
execute_process(COMMAND ptxas --version OUTPUT_VARIABLE PTXAS_OUT ERROR_QUIET)
if(PTXAS_OUT MATCHES "release ([0-9]+\\.[0-9]+)" AND CMAKE_MATCH_1 VERSION_LESS 12.3)
    set(MAX_REG_FLAG "-maxrregcount=24")
endif()

add_library(enhanced_tracer_tool SHARED
    tracer_tool/tracer_tool.cu
    tracer_tool/inject_funcs.cu
)

target_include_directories(enhanced_tracer_tool PUBLIC ${nvbit_SOURCE_DIR}/core)
target_link_directories(enhanced_tracer_tool PUBLIC ${nvbit_SOURCE_DIR}/core)
target_compile_definitions(enhanced_tracer_tool PRIVATE _FORCE_INLINES)
target_link_libraries(enhanced_tracer_tool PUBLIC nvbit CUDA::cudart_static CUDA::cuda_driver traces_enhanced)

set_target_properties(enhanced_tracer_tool PROPERTIES 
    POSITION_INDEPENDENT_CODE ON
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    PREFIX ""
)

# tracer_tool.cu: Standard compilation with no cloning
set_source_files_properties(tracer_tool/tracer_tool.cu PROPERTIES
    COMPILE_OPTIONS "-Xptxas;-cloning=no"
)

# inject_funcs.cu: Instrumentation injection flags
set_source_files_properties(tracer_tool/inject_funcs.cu PROPERTIES 
    COMPILE_OPTIONS "-Xptxas;-astoolspatch;--keep-device-functions;${MAX_REG_FLAG}"
)

