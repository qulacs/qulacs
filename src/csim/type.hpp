
/**
 * @file type.h
 * @brief basic definitins of types and macros
 */

#pragma once

#include <complex>

//! size_t for gcc
#include <stddef.h>

//! unsigned integer
using UINT = unsigned int;

//! complex value
using CTYPE = std::complex<double>;
using namespace std::complex_literals;
inline static double _cabs(CTYPE val) { return std::abs(val); }
inline static double _creal(CTYPE val) { return std::real(val); }
inline static double _cimag(CTYPE val) { return std::imag(val); }

//! dimension index
#ifdef _MSC_VER
// In MSVC, OpenMP only supports signed index
using ITYPE = signed long long;
#else
using ITYPE = unsigned long long;
#endif

//! check AVX2 support
#ifdef _MSC_VER
// MSVC
// In MSVC, flag __AVX2__ is not automatically set by default
#else
// GCC remove simd flag when AVX2 is not supported
#ifndef __AVX2__
#undef _USE_SIMD
#endif
#endif

//! define export command
#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility("default")))
#endif

//! ACLE
#ifdef _USE_SVE
#include "arm_acle.h"
#include "arm_sve.h"
#endif  // #ifdef _USE_SVE
