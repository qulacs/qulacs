#ifndef _UTIL_TYPE_H_
#define _UTIL_TYPE_H_

#ifndef PI
#define PI 3.141592653589793
#endif

#include <complex>
using CPPCTYPE = std::complex<double>;
using UINT = unsigned int;

#ifdef _MSC_VER
// In MSVC, OpenMP only supports signed index
using ITYPE = signed long long;
#else
using ITYPE = unsigned long long;
#endif

#endif  // #ifndef _UTIL_COMMON_CU_H_
