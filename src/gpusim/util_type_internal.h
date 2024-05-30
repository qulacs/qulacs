#ifndef _UTIL_TYPE_INTERNAL_H_
#define _UTIL_TYPE_INTERNAL_H_

#ifdef __HIP_PLATFORM_AMD__
#include <hip/hip_complex.h>
using GTYPE = hipDoubleComplex;
#else
#include <cuComplex.h>
using GTYPE = cuDoubleComplex;
#endif

#endif  // #ifndef _UTIL_COMMON_CU_H_
