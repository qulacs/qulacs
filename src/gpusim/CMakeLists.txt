cmake_minimum_required(VERSION 3.21)

file(GLOB GPUSIM_SRC
    "*.cu"
    "*.cuh"
    "*.h")

if (USE_GPU AND USE_HIP)
# ROCm HIP for AMD Instinct Accelerator
project(qulacs LANGUAGES HIP)

if (NOT DEFINED ROCM_PATH )
     set ( ROCM_PATH "/opt/rocm"  CACHE STRING "Default ROCM installation directory." )
endif ()

# Search for rocm in common locations
list(APPEND CMAKE_PREFIX_PATH ${ROCM_PATH}/lib/cmake ${ROCM_PATH})

# Find hip
find_package(hip REQUIRED)
find_package(hipblas REQUIRED)
find_package(hiprand REQUIRED)

set(CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_CXX_LINKER ${HIP_HIPCC_EXECUTABLE})
set(CMAKE_BUILD_TYPE Release)

message(STATUS "C++ Compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ Linker: ${CMAKE_CXX_LINKER}")

file(GLOB CUDA_FILES "*.cu")

add_library(gpusim_static STATIC ${GPUSIM_SRC})
set_target_properties(gpusim_static PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(gpusim_static PROPERTIES HIP_ARCHITECTURES "gfx90a;gfx942")
target_link_libraries(gpusim_static hip::host roc::hipblas hip::hiprand)
set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE HIP)

elseif(USE_GPU)
# CUDA for NVIDIA GPU

enable_language(CUDA)
if (MSVC)
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/wd 4819\">")
    add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/wd 4244\">")
    if (USE_OMP)
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler \"/openmp\">")
        add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>")
    endif()
endif()

find_package(CUDAToolkit REQUIRED)
message(STATUS "-- CUDA --")
message(STATUS "Toolkit Version: " ${CUDAToolkit_VERSION})
message(STATUS "Runtime Directory: " ${CUDAToolkit_LIBRARY_DIR})

add_library(gpusim_static STATIC ${GPUSIM_SRC})
add_library(gpusim_shared SHARED ${GPUSIM_SRC})

target_link_libraries(gpusim_static CUDA::cudart_static CUDA::curand_static CUDA::cublas_static)
target_link_libraries(gpusim_shared CUDA::cudart CUDA::curand CUDA::cublas)
set_source_files_properties(${CUDA_FILES} PROPERTIES LANGUAGE CUDA)

get_target_property(flags gpusim_static COMPILE_OPTIONS)
message(STATUS "Flag: " ${flags})

set_target_properties(gpusim_shared PROPERTIES EXCLUDE_FROM_ALL TRUE)

endif()
