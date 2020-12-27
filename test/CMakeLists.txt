cmake_minimum_required(VERSION 2.8)

add_subdirectory(csim)
add_subdirectory(cppsim)
add_subdirectory(vqcsim)

if (USE_GPU)
    add_subdirectory(gpusim)
endif()
