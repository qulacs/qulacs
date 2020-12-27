cmake_minimum_required(VERSION 2.8)

add_subdirectory(csim) 
if (USE_GPU)
    add_subdirectory(gpusim)
endif()
if (USE_MPI)
    add_subdirectory(mpisim)
endif()

add_subdirectory(cppsim)
add_subdirectory(vqcsim)

