#ifndef _UTIL_COMMON_CU_H_
#define _UTIL_COMMON_CU_H_

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuComplex.h>
//#include <cuda_runtime.h>
//#include <cuda.h>

#include <complex>

#define PI 3.141592653589793
//#define HOIST_GPU_DT
//#define CUDA_ERROR_CHECK
//#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )

typedef std::complex<double> CTYPE;
typedef cuDoubleComplex GTYPE;
typedef unsigned int UINT;
typedef unsigned long long ITYPE;

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility ("default")))
#endif

#endif // #ifndef _UTIL_COMMON_CU_H_