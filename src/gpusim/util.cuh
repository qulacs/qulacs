#ifndef _QCUDASIM_UTIL_CUH_
#define _QCUDASIM_UTIL_CUH_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <string>
#include <complex>
//#include <sys/time.h>
#include <cuComplex.h>
#include "util_type.h"
#include "util_type_internal.h"

//#include "util_type.h"

inline void checkCudaErrors(const cudaError error, std::string filename, int line)
{
	if (error != cudaSuccess){
		printf("Error: %s:%d, ", filename.c_str(), line);
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}

inline void checkCudaErrors(const cudaError error)
{
	if (error != cudaSuccess){
		printf("Error: %s:%d, ", __FILE__, __LINE__);
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}

//inline void memcpy_quantum_state_HostToDevice(CPPCTYPE* state_cpu, GTYPE* state_gpu, ITYPE dim){
inline void memcpy_quantum_state_HostToDevice(CPPCTYPE* state_cpu, GTYPE* state_gpu, ITYPE dim){
	checkCudaErrors(cudaSetDevice(0));
	checkCudaErrors(cudaMalloc((void**)&state_gpu, dim * sizeof(CPPCTYPE)));
	checkCudaErrors(cudaMemcpy(state_gpu, state_cpu, dim * sizeof(CPPCTYPE), cudaMemcpyHostToDevice));
}

/*
inline void print_quantum_state(GTYPE* state_gpu, ITYPE dim){
	CTYPE* state_cpu=(CTYPE*)malloc(sizeof(CTYPE)*dim);
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(state_cpu, state_gpu, dim * sizeof(CTYPE), cudaMemcpyDeviceToHost));
	for(int i=0;i<dim;++i){
		std::cout << i << " : " << state_cpu[i].real() << "+i" << state_cpu[i].imag() << '\n'; 
	}
	std::cout << '\n';
	free(state_cpu);
}
*/

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err)
	{
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n",
			file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif

	return;
}

inline __device__ ITYPE insert_zero_to_basis_index_device(ITYPE basis_index, unsigned int qubit_index){
    // ((basis_index >> qubit_index) << (qubit_index+1) )+ (basis_index % basis_mask)
	ITYPE temp_basis = (basis_index >> qubit_index) << (qubit_index+1);
    return (temp_basis + (basis_index & ( (1ULL<<qubit_index) -1)));
}

#endif // #ifndef _QCUDASIM_UTIL_CUH_
