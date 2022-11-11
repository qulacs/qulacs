#ifndef _UTIL_COMMON_CU_H_
#define _UTIL_COMMON_CU_H_

#include <cuComplex.h>

#include <complex>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define PI 3.141592653589793

using CTYPE = std::complex<double>;
using GTYPE = cuDoubleComplex;
using UINT = unsigned int;
using ITYPE = unsigned long long;

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility("default")))
#endif

#endif  // #ifndef _UTIL_COMMON_CU_H_