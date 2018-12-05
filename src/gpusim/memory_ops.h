#ifndef _MEMORY_OPS_CU_H_
#define _MEMORY_OPS_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_runtime.h>
//#include <cuda.h>
#include <cuComplex.h>
#include "util.h"
#include "util_common.h"
#include "util.cuh"

extern "C" DllExport __host__ void* allocate_quantum_state_host(ITYPE dim);
extern "C" DllExport __host__ void initialize_quantum_state_host(void* state, ITYPE dim);
extern "C" DllExport __host__ void release_quantum_state_host(void* state);
extern "C" DllExport __host__ void initialize_Haar_random_state_host(void *state, ITYPE dim);
extern "C" DllExport __host__ void initialize_Haar_random_state_with_seed_host(void *state, ITYPE dim, UINT seed);

#endif // _MEMORY_OPS_CU_H_

