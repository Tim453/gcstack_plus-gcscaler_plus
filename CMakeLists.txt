cmake_minimum_required(VERSION 3.28)
project(GCStack+GCScaler)


add_subdirectory(gcstack+)
add_subdirectory(gcscaler+)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
