#pragma once

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <complex>
#include <csim/type.hpp>
using CPPCTYPE = std::complex<double>;
using ComplexVector = Eigen::VectorXcd;

// In order to use matrix raw-data without reordering, we use RowMajor as
// default.
using ComplexMatrix =
    Eigen::Matrix<CPPCTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using SparseComplexMatrix = Eigen::SparseMatrix<CPPCTYPE>;
using SparseComplexMatrixRowMajor =
    Eigen::SparseMatrix<CPPCTYPE, Eigen::RowMajor>;

#ifdef __GNUC__

#if __GNUC__ >= 8
using namespace std::complex_literals;
#endif
#endif

#ifdef _MSC_VER
inline static FILE* popen(const char* command, const char* mode) {
    return _popen(command, mode);
}
inline static void pclose(FILE* fp) { _pclose(fp); }
#define strncasecmp _strnicmp
#define strcasecmp _stricmp
#endif

#if defined(__MINGW32__) || defined(_MSC_VER)
#define DllExport __declspec(dllexport)
#else
#define DllExport __attribute__((visibility("default")))
#endif
