#include <assert.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "gpu_wrapping.h"
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
    if (device_number != current_device) gpuSetDevice(device_number);
    gpuStream_t* stream =
        (gpuStream_t*)malloc(max_cuda_stream * sizeof(gpuStream_t));
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        gpuStreamCreate(&stream[i]);
    void* gpu_stream = reinterpret_cast<void*>(stream);
    return gpu_stream;
}

__host__ void release_cuda_stream_host(void* gpu_stream,
    unsigned int max_cuda_stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);
    gpuStream_t* stream = reinterpret_cast<gpuStream_t*>(gpu_stream);
    for (unsigned int i = 0; i < max_cuda_stream; ++i)
        gpuStreamDestroy(stream[i]);
    free(stream);
}

__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < dim) {
        state_gpu[idx] = make_gpuDoubleComplex(0.0, 0.0);
    }
    if (idx == 0) state_gpu[idx] = make_gpuDoubleComplex(1.0, 0.0);
}

__host__ void* allocate_quantum_state_host(
    ITYPE dim, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);
    GTYPE* state_gpu;
    checkGpuErrors(
        gpuMalloc((void**)&state_gpu, dim * sizeof(GTYPE)), __FILE__, __LINE__);
    void* psi_gpu = reinterpret_cast<void*>(state_gpu);
    return psi_gpu;
}

__host__ void initialize_quantum_state_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);

    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(init_qstate);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;
    init_qstate<<<grid, block, 0, *gpu_stream>>>(state_gpu, dim);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);
}

__host__ void release_quantum_state_host(
    void* state, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);
    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    checkGpuErrors(gpuFree(state_gpu), __FILE__, __LINE__);
}

__global__ void init_rnd(
    gpurandState* const rnd_state, const unsigned int seed, ITYPE dim) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < dim) {
        gpurand_init(seed, tid, 0, &rnd_state[tid]);
    }
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
    gpurandState* rnd_state, GTYPE* state, ITYPE dim) {
    ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
    // double2 rnd;
    double tmp1, tmp2;
    double real, imag;
    gpurandState localState = rnd_state[idx];
    if (idx < dim) {
        // rnd = curand_normal2_double(&localState);
        tmp1 = gpurand_uniform_double(&localState);
        tmp2 = gpurand_uniform_double(&localState);
        real = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        tmp1 = gpurand_uniform_double(&localState);
        tmp2 = gpurand_uniform_double(&localState);
        imag = sqrt(-1.0 * log(tmp1)) * sinpi(2.0 * tmp2);
        state[idx] = make_gpuDoubleComplex(real, imag);
        rnd_state[idx] = localState;
    }
}

__host__ void initialize_Haar_random_state_with_seed_host(void* state,
    ITYPE dim, UINT seed, void* stream, unsigned int device_number) {
    int current_device = get_current_device();
    if (device_number != current_device) gpuSetDevice(device_number);

    GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
    gpuStream_t* gpu_stream = reinterpret_cast<gpuStream_t*>(stream);
    // const ITYPE ignore_first = 40;
    double norm = 0.;

    gpurandState* rnd_state;
    checkGpuErrors(gpuMalloc((void**)&rnd_state, dim * sizeof(gpurandState)),
        __FILE__, __LINE__);

    // CURAND_RNG_PSEUDO_XORWOW
    // CURAND_RNG_PSEUDO_MT19937 offset cannot be used and need sm_35 or higher.

    unsigned int max_block_size =
        get_block_size_to_maximize_occupancy(init_rnd);
    unsigned int block = dim <= max_block_size ? dim : max_block_size;
    unsigned int grid = (dim + block - 1) / block;

    init_rnd<<<grid, block, 0, *gpu_stream>>>(rnd_state, seed, dim);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);

    rand_normal_xorwow<<<grid, block, 0, *gpu_stream>>>(
        rnd_state, state_gpu, dim);
    checkGpuErrors(gpuGetLastError(), __FILE__, __LINE__);

    checkGpuErrors(gpuStreamSynchronize(*gpu_stream), __FILE__, __LINE__);
    checkGpuErrors(gpuFree(rnd_state), __FILE__, __LINE__);
    state = reinterpret_cast<void*>(state_gpu);

    norm = state_norm_squared_host(state, dim, gpu_stream, device_number);
    normalize_host(norm, state, dim, gpu_stream, device_number);
}

__host__ void initialize_Haar_random_state_host(
    void* state, ITYPE dim, void* stream, unsigned int device_number) {
    initialize_Haar_random_state_with_seed_host(
        state, dim, (unsigned)time(NULL), stream, device_number);
}
