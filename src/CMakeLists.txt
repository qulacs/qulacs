cmake_minimum_required(VERSION 2.8)

if(MSVC)
else()
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_C}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_CPP}")    
endif()

add_subdirectory(csim)
if(USE_GPU)
	add_subdirectory(gpusim)
endif()
add_subdirectory(cppsim)
add_subdirectory(vqcsim)
