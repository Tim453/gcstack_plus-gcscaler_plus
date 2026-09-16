project(GCStack)



find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams serialization)

# Enhanced traces and gcstack
add_subdirectory(util/traces_enhanced)
add_subdirectory(gpu-simulator)
add_subdirectory(util/tracer_nvbit)
