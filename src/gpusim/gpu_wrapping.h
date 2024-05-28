#ifndef _GPU_WRAPPING_H_
#define _GPU_WRAPPING_H_

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_complex.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

using gpuStream_t = hipStream_t;
using gpuError_t = hipError_t;
using gpurandState = hiprandState;
using gpublasStatus_t = hipblasStatus_t;
using gpublasHandle_t = hipblasHandle_t;
using gpublasDoubleComplex = hipblasDoubleComplex;
#define gpuSetDevice hipSetDevice
#define gpuGetDevice hipGetDevice
#define gpuGetDeviceCount hipGetDeviceCount
#define gpuStreamCreate hipStreamCreate
#define gpuStreamDestroy hipStreamDestroy
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuMalloc hipMalloc
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemcpyToSymbol hipMemcpyToSymbol
#define gpuMemcpyToSymbolAsync hipMemcpyToSymbolAsync
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemsetAsync hipMemsetAsync
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuFree hipFree
#define gpuSuccess hipSuccess
#define gpuGetLastError hipGetLastError
#define make_gpuDoubleComplex make_hipDoubleComplex
#define gpurand_init hiprand_init
#define gpurand_uniform_double hiprand_uniform_double
#define gpublasCreate hipblasCreate
#define gpublasSetVector hipblasSetVector
#define gpublasSetMatrix hipblasSetMatrix
#define gpublasSetStream hipblasSetStream
#define gpublasZgemv hipblasZgemv
#define gpublasZgemm hipblasZgemm
#define gpublasDznrm2 hipblasDznrm2
#define gpublasZdotc hipblasZdotc
#define gpublasGetVector hipblasGetVector
#define gpublasGetMatrix hipblasGetMatrix
#define gpublasDestroy hipblasDestroy
#define gpuCabs hipCabs
#define gpuCadd hipCadd
#define gpuCmul hipCmul
#define gpuConj hipConj
#define gpuCreal hipCreal
#define gpuCimag hipCimag
#define GPUBLAS_OP_T HIPBLAS_OP_T
#define GPUBLAS_STATUS_SUCCESS HIPBLAS_STATUS_SUCCESS
#define GPU_SYMBOL HIP_SYMBOL

#else

#include <cuComplex.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "device_launch_parameters.h"

using gpuStream_t = cudaStream_t;
using gpuError_t = cudaError_t;
using gpurandState = curandState;
using gpublasStatus_t = cublasStatus_t;
using gpublasHandle_t = cublasHandle_t;
using gpublasDoubleComplex = cuDoubleComplex;
#define gpuSetDevice cudaSetDevice
#define gpuGetDevice cudaGetDevice
#define gpuGetDeviceCount cudaGetDeviceCount
#define gpuStreamCreate cudaStreamCreate
#define gpuStreamDestroy cudaStreamDestroy
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuMalloc cudaMalloc
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemcpyToSymbol cudaMemcpyToSymbol
#define gpuMemcpyToSymbolAsync cudaMemcpyToSymbolAsync
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemsetAsync cudaMemsetAsync
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuFree cudaFree
#define gpuSuccess cudaSuccess
#define gpuGetLastError cudaGetLastError
#define make_gpuDoubleComplex make_cuDoubleComplex
#define gpurand_init curand_init
#define gpurand_uniform_double curand_uniform_double
#define gpublasCreate cublasCreate
#define gpublasSetVector cublasSetVector
#define gpublasSetMatrix cublasSetMatrix
#define gpublasZgemv cublasZgemv
#define gpublasZgemm cublasZgemm
#define gpublasDznrm2 cublasDznrm2
#define gpublasZdotc cublasZdotc
#define gpublasGetVector cublasGetVector
#define gpublasGetMatrix cublasGetMatrix
#define gpublasSetStream cublasSetStream
#define gpublasDestroy cublasDestroy
#define gpuCabs cuCabs
#define gpuCadd cuCadd
#define gpuCmul cuCmul
#define gpuConj cuConj
#define gpuCreal cuCreal
#define gpuCimag cuCimag
#define GPUBLAS_OP_T CUBLAS_OP_T
#define GPUBLAS_STATUS_SUCCESS CUBLAS_STATUS_SUCCESS
#define GPU_SYMBOL

#endif

#endif  // _GPU_WRAPPING_H_
