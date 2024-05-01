#ifndef _GPU_WRAPPING_H_
#define _GPU_WRAPPING_H_ 

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>

using gpuStream_t = hipStream_t;
#define gpuSetDevice hipSetDevice
#define gpuStreamCreate hipStreamCreate

#else

#include <cuda_runtime.h>
#include "device_launch_parameters.h"

using gpuStream_t = cudaStream_t;
#define gpuSetDevice cudaSetDevice
#define gpuStreamCreate cudaStreamCreate

#endif

#endif  // _GPU_WRAPPING_H_ 
