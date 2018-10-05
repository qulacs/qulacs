
/**
 * @file type.h
 * @brief basic definitins of types
 */


#pragma once

// When csim is compiled with C++, std::complex<double> is used instead of double _Complex
#ifdef _MSC_VER
#include <complex>
#else
#include <complex.h>
#endif

//! unsigned integer
typedef unsigned int UINT;

//! complex value
#ifdef _MSC_VER
typedef std::complex<double> CTYPE;
using namespace std::complex_literals;
inline static double cabs(CTYPE val) { return std::abs(val); }
inline static double creal(CTYPE val) { return std::real(val); }
inline static double cimag(CTYPE val) { return std::imag(val); }
#else
typedef double _Complex CTYPE;
#endif

//! dimension index
#ifdef _MSC_VER
// In MSVC, OpenMP only supports signed index
typedef signed long long ITYPE;
#else
typedef unsigned long long ITYPE;
#endif


#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility ("default")))
#endif
