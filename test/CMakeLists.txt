cmake_minimum_required(VERSION 3.0)

add_subdirectory(csim)
add_subdirectory(cppsim)
if(USE_GPU)
	add_subdirectory(gpusim)
endif()
add_subdirectory(vqcsim)
