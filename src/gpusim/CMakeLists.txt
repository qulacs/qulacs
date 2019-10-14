cmake_minimum_required(VERSION 2.8)

find_package(CUDA REQUIRED)
if(MSVC)
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4819\"")
	list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4244\"")
	if(USE_OMP)
		list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/openmp\"")
		list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")
	endif()
endif()

file(GLOB GPUSIM_SRC
    "*.cu"
    "*.cuh"
    "*.h"
)

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

#CUDA_GENERATED_OUTPUT_DIR(CMAKE_CURRENT_BINARY_DIR/build)
#CUDA_curand_LIBRARY
#cmake --help-module FindCUDA

CUDA_COMPILE(gpusim_static ${GPUSIM_SRC})
CUDA_COMPILE(gpusim_shared ${GPUSIM_SRC})

CUDA_ADD_LIBRARY(gpusim_static ${GPUSIM_SRC} STATIC)
CUDA_ADD_LIBRARY(gpusim_shared ${GPUSIM_SRC} SHARED)

target_link_libraries(gpusim_static ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
target_link_libraries(gpusim_shared ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})

message("Flag: " ${CUDA_NVCC_FLAGS})

set_target_properties(gpusim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)
