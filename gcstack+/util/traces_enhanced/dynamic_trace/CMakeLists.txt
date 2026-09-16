




add_library(proto_objects STATIC)


protobuf_generate(
    TARGET proto_objects
    LANGUAGE cpp
    DESCRIPTORS
    PROTOS 
    address.proto
    dim3d.proto
    gpu_device.proto
    instruction.proto
    kernel.proto
    threadblock.proto
    trace.proto
    warp.proto
    cuda_stream.proto
)


target_link_libraries(proto_objects PUBLIC protobuf::libprotobuf)
target_include_directories(proto_objects PUBLIC ${CMAKE_CURRENT_BINARY_DIR})