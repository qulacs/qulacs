cmake_minimum_required(VERSION 3.0)

file(GLOB GPUSIM_SRC
    "*.cu"
    "*.cuh"
    "*.h")

if (USE_GPU AND USE_HIP)
# ROCm HIP for AMD Instinct Accelerator

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/hip ${ROCM_PATH})

# Find hip
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(rocrand REQUIRED)

#file(GLOB GPUSIM_FILES
#    "*.cpp"
#    "*.cuh"
#    "*.h")
#message("-- ROCm --")
#message("Version: " ${HIP_VERSION})

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

add_library(gpusim_static STATIC ${GPUSIM_FILES})
target_link_libraries(gpusim_static hip::host roc::hipblas roc::rocrand)

message("Flag: " ${HIP_HIPCC_FLAGS})
message(FATAL_ERROR "Not Ready yet")

elseif(USE_GPU)
# CUDA for NVIDIA GPU

find_package(CUDA REQUIRED)
if (MSVC)
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4819\"")
    list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/wd 4244\"")
    if (USE_OMP)
        list(APPEND CUDA_NVCC_FLAGS "-Xcompiler \"/openmp\"")
        list(APPEND CUDA_NVCC_FLAGS "--default-stream per-thread")
    endif()
endif()

message("-- CUDA --")
message("Version: " ${CUDA_VERSION})
message("Library: " ${CUDA_CUDA_LIBRARY})
message("Runtime: " ${CUDA_CUDART_LIBRARY})

#CUDA_GENERATED_OUTPUT_DIR(CMAKE_CURRENT_BINARY_DIR / build)
#CUDA_curand_LIBRARY
#cmake-- help - module FindCUDA

CUDA_COMPILE(gpusim_static ${GPUSIM_SRC})
CUDA_COMPILE(gpusim_shared ${GPUSIM_SRC})

CUDA_ADD_LIBRARY(gpusim_static ${GPUSIM_SRC} STATIC)
CUDA_ADD_LIBRARY(gpusim_shared ${GPUSIM_SRC} SHARED)

target_link_libraries(gpusim_static ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})
target_link_libraries(gpusim_shared ${CUDA_curand_LIBRARY} ${CUDA_cublas_LIBRARY})

message("Flag: " ${CUDA_NVCC_FLAGS})

set_target_properties(gpusim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)

endif()
