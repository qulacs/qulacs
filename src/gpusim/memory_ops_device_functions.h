#ifndef _MEMORY_OPS_CU_DEVICE_H_
#define _MEMORY_OPS_CU_DEVICE_H_

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <hiprand/hiprand.h>
#include <hiprand/hiprand_kernel.h>

#else

#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#endif

#include "util_export.h"
#include "util_type.h"
#include "util_type_internal.h"

__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim);
#ifdef __HIP_PLATFORM_AMD__
__global__ void init_rnd(
    hiprandState* const rnd_state, const unsigned int seed, ITYPE dim);
__global__ void rand_normal_xorwow(
    hiprandState* rnd_state, GTYPE* state, ITYPE dim);
#else
__global__ void init_rnd(
    curandState* const rnd_state, const unsigned int seed, ITYPE dim);
__global__ void rand_normal_xorwow(
    curandState* rnd_state, GTYPE* state, ITYPE dim);
#endif

#endif  // _MEMORY_OPS_CU_DEVICE_H_
