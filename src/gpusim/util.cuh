#ifndef _QCUDASIM_UTIL_CUH_
#define _QCUDASIM_UTIL_CUH_

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_complex.h>

#include "hip/hip_runtime.h"
#else
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <sys/time.h>
#include <cuComplex.h>
#endif

#include "util_type.h"
#include "util_type_internal.h"

// #include "util_type.h"

inline void checkGpuErrors(
#ifdef __HIP_PLATFORM_AMD__
    const hipError_t error, std::string filename, int line) {
    if (error != hipSuccess) {
        printf("Error: %s:%d, ", filename.c_str(), line);
        printf("code: %d, reason: %s\n", error, hipGetErrorString(error));
#else
    const cudaError error, std::string filename, int line) {
    if (error != cudaSuccess) {
        printf("Error: %s:%d, ", filename.c_str(), line);
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
#endif
        exit(1);
    }
}

// As a result of the experience, using `cudaOccupancyMaxPotentialBlockSize`
// does not lose performance. ref:
// https://github.com/qulacs/qulacs/issues/618#issuecomment-2011658886
#ifdef __HIP_PLATFORM_AMD__
#define get_block_size_to_maximize_occupancy(x)      \
    ({                                               \
        int min_grid_size, block_size;               \
        hipOccupancyMaxPotentialBlockSize(           \
            &min_grid_size, &block_size, (x), 0, 0); \
        block_size;                                  \
    })
#else
template <typename F>
inline unsigned int get_block_size_to_maximize_occupancy(F func,
    unsigned int dynamic_s_mem_size = 0, unsigned int block_size_limit = 0) {
    int block_size, min_grid_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, func,
        dynamic_s_mem_size, block_size_limit);
    return block_size;
}
#endif

/*
//inline void memcpy_quantum_state_HostToDevice(CPPCTYPE* state_cpu, GTYPE*
state_gpu, ITYPE dim){ inline void memcpy_quantum_state_HostToDevice(CPPCTYPE*
state_cpu, GTYPE* state_gpu, ITYPE dim, void* stream, UINT device_number){
        cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
        int current_device = get_current_device();
        if (device_number != current_device) cudaSetDevice(device_number);

        checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CPPCTYPE)),
__FILE__, __LINE__); checkCudaErrors(cudaMemcpyAsync(state_gpu, state_cpu, dim *
sizeof(CPPCTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
}

inline void memcpy_quantum_state_HostToDevice(CPPCTYPE* state_cpu, GTYPE*
state_gpu, ITYPE dim, void* stream, unsigned int device_number){ cudaStream_t*
cuda_stream = reinterpret_cast<cudaStream_t*>(stream); int current_device =
get_current_device(); if (device_number != current_device)
cudaSetDevice(device_number);

        checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CPPCTYPE)),
__FILE__, __LINE__); checkCudaErrors(cudaMemcpyAsync(state_gpu, state_cpu, dim *
sizeof(CPPCTYPE), cudaMemcpyHostToDevice, *cuda_stream), __FILE__, __LINE__);
}
*/
#ifdef __HIP_PLATFORM_AMD__
inline void __cudaSafeCall(hipError_t err, const char* file, const int line) {
#else
inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
#endif
#ifdef CUDA_ERROR_CHECK
#ifdef __HIP_PLATFORM_AMD__
    if (hipSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            hipGetErrorString(err));
#else
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
            cudaGetErrorString(err));
#endif
        exit(-1);
    }
#endif

    return;
}

inline __device__ ITYPE insert_zero_to_basis_index_device(
    ITYPE basis_index, unsigned int qubit_index) {
    // ((basis_index >> qubit_index) << (qubit_index+1) )+ (basis_index %
    // basis_mask)
    ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index + 1);
    return (temp_basis + (basis_index & ((1ULL << qubit_index) - 1)));
}

#endif  // #ifndef _QCUDASIM_UTIL_CUH_
