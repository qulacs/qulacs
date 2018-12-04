#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>
// for using cublas 
#include <cublas_v2.h>

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <complex>
#include <assert.h>
#include <algorithm>
#include <cuComplex.h>
#include "util.h"
#include "util_common.h"
#include "util.cuh"
#include "memory_ops.h"

__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim){
	ITYPE idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < dim) {
		state_gpu[idx] = make_cuDoubleComplex(0.0, 0.0);
	}
	if (idx == 0) state_gpu[idx] = make_cuDoubleComplex(1.0, 0.0);
}

// void* (GTYPE*)
__host__ void* allocate_quantum_state_host(ITYPE dim){
	GTYPE *state_gpu;
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(GTYPE)));
	void* psi_gpu = reinterpret_cast<void*>(state_gpu);
    return psi_gpu;
}

__host__ void initialize_quantum_state_host(void* state, ITYPE dim){
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	cudaError cudaStatus;
	unsigned int block = dim <= 1024 ? dim : 1024;
	unsigned int grid = dim / block;
	init_qstate << <grid, block >> >(state_gpu, dim);

    checkCudaErrors(cudaDeviceSynchronize(), __FILE__, __LINE__);
	cudaStatus = cudaGetLastError();
	checkCudaErrors(cudaStatus, __FILE__, __LINE__);
	state = reinterpret_cast<void*>(state_gpu);
}

__host__ void release_quantum_state_host(void* state){
	GTYPE* state_gpu = reinterpret_cast<GTYPE*>(state);
	checkCudaErrors(cudaFree(state_gpu), __FILE__, __LINE__);
}

