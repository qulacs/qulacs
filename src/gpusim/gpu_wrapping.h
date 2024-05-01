#ifndef _GPU_WRAPPING_H_
#define _GPU_WRAPPING_H_ 

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

using gpuStream_t = hipStream_t;
using gpurandState = hiprandState;
#define gpuSetDevice hipSetDevice
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define make_gpuDoubleComplex make_hipDoubleComplex
#define gpuMalloc hipMalloc
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuGetLastError hipGetLastError
#define gpuFree hipFree
#define gpurand_init hiprand_init
#define gpurand_uniform_double hiprand_uniform_double

#else

#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuComplex.h>
#include <curand.h>
#include <curand_kernel.h>

using gpuStream_t = cudaStream_t;
using gpurandState = curandState;
#define gpuSetDevice cudaSetDevice
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define make_gpuDoubleComplex make_cuDoubleComplex
#define gpuMalloc cudaMalloc
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuFree cudaFree
#define gpurand_init curand_init
#define gpurand_uniform_double curand_uniform_double

#endif

#endif  // _GPU_WRAPPING_H_ 
