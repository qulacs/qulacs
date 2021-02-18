#pragma once

#ifndef _MSC_VER
extern "C" {
#include <csim/type.h>
}
#else
#include <csim/type.h>
#endif

#include <cereal/cereal.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/archives/binary.hpp>

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <complex>
typedef std::complex<double> CPPCTYPE;
typedef Eigen::VectorXcd ComplexVector;

// In order to use matrix raw-data without reordering, we use RowMajor as
// default.
typedef Eigen::Matrix<CPPCTYPE, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    ComplexMatrix;
typedef Eigen::SparseMatrix<CPPCTYPE> SparseComplexMatrix;

// reference:
// https://stackoverflow.com/questions/22884216/serializing-eigenmatrix-using-cereal-library

namespace cereal {
template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options>
inline typename std::enable_if<
    traits::is_output_serializable<BinaryData<_Scalar>, Archive>::value,
    void>::type
save(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options> const& m) {
    int32_t rows = m.rows();
    int32_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
}

template <class Archive, class _Scalar, int _Rows, int _Cols, int _Options>
inline typename std::enable_if<
    traits::is_input_serializable<BinaryData<_Scalar>, Archive>::value,
    void>::type
load(Archive& ar, Eigen::Matrix<_Scalar, _Rows, _Cols, _Options>& m) {
    int32_t rows;
    int32_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(_Scalar))));
}
}  // namespace cereal

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
