#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <assert.h>
#include <hip/hip_complex.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#else

#include <cuda_runtime.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
// #include <cuda.h>
#include <assert.h>
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>

#endif

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "memory_ops.h"
#include "memory_ops_device_functions.h"
#include "stat_ops.h"
#include "update_ops_cuda.h"
#include "util.cuh"
#include "util_func.h"
#include "util_type.h"
#include "util_type_internal.h"

__host__ void* allocate_cuda_stream_host(
    unsigned int max_cuda_stream, unsigned int device_number) {
    int current_device = get_current_device();
#ifdef __HIP_PLATFORM_AMD__
    if (device_number != current_device) hipSetDevice(device_number);
    hipStream_t* stream =
        (hipStream_t*)malloc(max_cuda_stream * sizeof(hipStream_t));
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        hipStreamCreate(&stream[i]);
    void* hip_stream = reinterpret_cast<void*>(stream);
    return hip_stream;
#else
    if (device_number != current_device) cudaSetDevice(device_number);
    cudaStream_t* stream =
        (cudaStream_t*)malloc(max_cuda_stream * sizeof(cudaStream_t));
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        cudaStreamCreate(&stream[i]);
    void* cuda_stream = reinterpret_cast<void*>(stream);
    return cuda_stream;
#endif
}

#ifdef __HIP_PLATFORM_AMD__
__host__ void release_cuda_stream_host(void* hip_stream,
    unsigned int max_cuda_stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) hipSetDevice(device_number);
    hipStream_t* stream = reinterpret_cast<hipStream_t*>(hip_stream);
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        hipStreamDestroy(stream[i]);
    free(stream);
}
#else
__host__ void release_cuda_stream_host(void* cuda_stream,
    unsigned int max_cuda_stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) cudaSetDevice(device_number);
    cudaStream_t* stream = reinterpret_cast<cudaStream_t*>(cuda_stream);
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        cudaStreamDestroy(stream[i]);
    free(stream);
}
#endif

__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
#ifdef __HIP_PLATFORM_AMD__
    if (idx < dim) {
        state_gpu[idx] = make_hipDoubleComplex(0.0, 0.0);
    }
    if (idx == 0) state_gpu[idx] = make_hipDoubleComplex(1.0, 0.0);
#else
    if (idx < dim) {
        state_gpu[idx] = make_cuDoubleComplex(0.0, 0.0);
    }
    if (idx == 0) state_gpu[idx] = make_cuDoubleComplex(1.0, 0.0);
#endif
}

__host__ void* allocate_quantum_state_host(
    ITYPE dim, unsigned int device_number) {
    int current_device = get_current_device();
#ifdef __HIP_PLATFORM_AMD__
    if (device_number != current_device) hipSetDevice(device_number);
    GTYPE* state_gpu;
    checkCudaErrors(hipMalloc((void**)&state_gpu, dim * sizeof(GTYPE)),
        __FILE__, __LINE__);
#else
    if (device_number != current_device) cudaSetDevice(device_number);
    GTYPE* state_gpu;
    checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(GTYPE)),
        __FILE__, __LINE__);
#endif
    void* psi_gpu = reinterpret_cast<void*>(state_gpu);
    return psi_gpu;
}

__host__ void initialize_quantum_state_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
#ifdef __HIP_PLATFORM_AMD__
    if (device_number != current_device) hipSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    hipStream_t* hip_stream = reinterpret_cast<hipStream_t*>(stream);
#else
    if (device_number != current_device) cudaSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
#endif

    unsigned int block = dim <= 1024 ? dim : 1024;
    unsigned int grid = dim / block;
#ifdef __HIP_PLATFORM_AMD__
    init_qstate<<<grid, block, 0, *hip_stream>>>(state_gpu, dim);

    checkCudaErrors(hipStreamSynchronize(*hip_stream), __FILE__, __LINE__);
    checkCudaErrors(hipGetLastError(), __FILE__, __LINE__);
#else
    init_qstate<<<grid, block, 0, *cuda_stream>>>(state_gpu, dim);

    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);
#endif
}

__host__ void release_quantum_state_host(
    void* state, unsigned int device_number) {
    int current_device = get_current_device();
#ifdef __HIP_PLATFORM_AMD__
    if (device_number != current_device) hipSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    checkCudaErrors(hipFree(state_gpu), __FILE__, __LINE__);
#else
    if (device_number != current_device) cudaSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    checkCudaErrors(cudaFree(state_gpu), __FILE__, __LINE__);
#endif
}

__global__ void init_rnd(
#ifdef __HIP_PLATFORM_AMD__
    hiprandState* const rnd_state, const unsigned int seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed, tid, 0, &rnd_state[tid]);
#else
    curandState* const rnd_state, const unsigned int seed) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curand_init(seed, tid, 0, &rnd_state[tid]);
#endif
}

/*
__global__ void rand_normal_mtgp32(curandState* rnd_state, GTYPE* state, ITYPE
dim){ ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x; double2 rnd;
    curandStateMtgp32 localState = rnd_state[idx];
        if (idx < dim) {
        rnd = curand_normal2_double(&localState);
        state[idx] = make_cuDoubleComplex(rnd.x, rnd.y);
        rnd_state[idx] = localState;
    }
}
*/

__global__ void rand_normal_xorwow(
#ifdef __HIP_PLATFORM_AMD__
    hiprandState* rnd_state, GTYPE* state, ITYPE dim) {
#else
    curandState* rnd_state, GTYPE* state, ITYPE dim) {
#endif
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    // double2 rnd;
    double tmp1, tmp2;
    double real, imag;
#ifdef __HIP_PLATFORM_AMD__
    hiprandState localState = rnd_state[idx];
#else
    curandStateXORWOW localState = rnd_state[idx];
#endif
    if (idx < dim) {
        // rnd = curand_normal2_double(&localState);
#ifdef __HIP_PLATFORM_AMD__
        tmp1 = hiprand_uniform_double(&localState);
        tmp2 = hiprand_uniform_double(&localState);
        real = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        tmp1 = hiprand_uniform_double(&localState);
        tmp2 = hiprand_uniform_double(&localState);
        imag = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        state[idx] = make_hipDoubleComplex(real, imag);
#else
        tmp1 = curand_uniform_double(&localState);
        tmp2 = curand_uniform_double(&localState);
        real = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        tmp1 = curand_uniform_double(&localState);
        tmp2 = curand_uniform_double(&localState);
        imag = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        state[idx] = make_cuDoubleComplex(real, imag);
#endif
        rnd_state[idx] = localState;
    }
}

__host__ void initialize_Haar_random_state_with_seed_host(void* state,
    ITYPE dim, UINT seed, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
#ifdef __HIP_PLATFORM_AMD__
    if (device_number != current_device) hipSetDevice(device_number);
#else
    if (device_number != current_device) cudaSetDevice(device_number);
#endif

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
#ifdef __HIP_PLATFORM_AMD__
    hipStream_t* hip_stream = reinterpret_cast<hipStream_t*>(stream);
#else
    cudaStream_t* cuda_stream = reinterpret_cast<cudaStream_t*>(stream);
#endif
    // const ITYPE ignore_first = 40;
    double norm = 0.;

#ifdef __HIP_PLATFORM_AMD__
    hiprandState* rnd_state;
    checkCudaErrors(hipMalloc((void**)&rnd_state, dim * sizeof(hiprandState)),
        __FILE__, __LINE__);
#else
    curandState* rnd_state;
    checkCudaErrors(cudaMalloc((void**)&rnd_state, dim * sizeof(curandState)),
        __FILE__, __LINE__);
#endif

    // CURAND_RNG_PSEUDO_XORWOW
    // CURAND_RNG_PSEUDO_MT19937 offset cannot be used and need sm_35 or higher.

    unsigned int block = dim <= 512 ? dim : 512;
    unsigned int grid = dim / block;

#ifdef __HIP_PLATFORM_AMD__
    init_rnd<<<grid, block, 0, *hip_stream>>>(rnd_state, seed);
    checkCudaErrors(hipGetLastError(), __FILE__, __LINE__);

    rand_normal_xorwow<<<grid, block, 0, *hip_stream>>>(
        rnd_state, state_gpu, dim);
    checkCudaErrors(hipGetLastError(), __FILE__, __LINE__);

    checkCudaErrors(hipStreamSynchronize(*hip_stream), __FILE__, __LINE__);
    checkCudaErrors(hipFree(rnd_state), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);

    norm = state_norm_squared_host(state, dim, hip_stream, device_number);
    normalize_host(norm, state, dim, hip_stream, device_number);
#else
    init_rnd<<<grid, block, 0, *cuda_stream>>>(rnd_state, seed);
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);

    rand_normal_xorwow<<<grid, block, 0, *cuda_stream>>>(
        rnd_state, state_gpu, dim);
    checkCudaErrors(cudaGetLastError(), __FILE__, __LINE__);

    checkCudaErrors(cudaStreamSynchronize(*cuda_stream), __FILE__, __LINE__);
    checkCudaErrors(cudaFree(rnd_state), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);

    norm = state_norm_squared_host(state, dim, cuda_stream, device_number);
    normalize_host(norm, state, dim, cuda_stream, device_number);
#endif
}

__host__ void initialize_Haar_random_state_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    initialize_Haar_random_state_with_seed_host(
        state, dim, (unsigned)time(NULL), stream, device_number);
}
