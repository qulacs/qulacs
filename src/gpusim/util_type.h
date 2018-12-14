#ifndef _UTIL_TYPE_H_
#define _UTIL_TYPE_H_

#ifndef PI
#include <complex>
#define PI 3.141592653589793
typedef std::complex<double> CPPCTYPE;
typedef unsigned int UINT;
#ifdef _MSC_VER
// In MSVC, OpenMP only supports signed index
typedef signed long long ITYPE;
#else
typedef unsigned long long ITYPE;
#endif
#endif

#endif // #ifndef _UTIL_COMMON_CU_H_