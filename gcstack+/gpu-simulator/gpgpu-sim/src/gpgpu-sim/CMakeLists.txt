file(GLOB SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/remodeling/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/remodeling/fusedMemory/*.cc
)

if(NOT GPGPUSIM_POWER_MODEL)
    list(REMOVE_ITEM SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/power_interface.cc)
endif()

add_library(gpu_uarch_sim OBJECT 
    ${SOURCES}
)

target_link_libraries(gpu_uarch_sim PUBLIC
    traces_enhanced
    OpenMP::OpenMP_CXX
    OpenMP::OpenMP_C
)

if(GPGPUSIM_POWER_MODEL)
    target_include_directories(gpu_uarch_sim PRIVATE ${CMAKE_SOURCE_DIR}/src/accelwattch)
endif()