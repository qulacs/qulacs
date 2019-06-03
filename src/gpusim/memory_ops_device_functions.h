#ifndef _MEMORY_OPS_CU_DEVICE_H_
#define _MEMORY_OPS_CU_DEVICE_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>

#include "util_export.h"
#include "util_type.h"
#include "util_type_internal.h"


__global__ void init_qstate(GTYPE* state_gpu, ITYPE dim);
__global__ void init_rnd(curandState *const rnd_state, const unsigned int seed);
__global__ void rand_normal_xorwow(curandState* rnd_state, GTYPE* state, ITYPE dim);


#endif // _MEMORY_OPS_CU_DEVICE_H_

