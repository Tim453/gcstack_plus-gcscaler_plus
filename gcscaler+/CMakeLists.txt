cmake_minimum_required(VERSION 3.24.0)
project(GCom VERSION 0.1.0 LANGUAGES C CXX)


set(BOOST_INCLUDE_LIBRARIES serialization)
set(BOOST_ENABLE_CMAKE ON)

add_subdirectory(third-party)
add_subdirectory(src)
