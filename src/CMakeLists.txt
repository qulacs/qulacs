cmake_minimum_required(VERSION 3.0)

# add warning flags
if(NOT MSVC)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${WARNING_C}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${WARNING_CPP}")
endif()

add_subdirectory(csim)
if (USE_GPU)
    add_subdirectory(gpusim)
endif()

add_subdirectory(cppsim)
add_subdirectory(vqcsim)
