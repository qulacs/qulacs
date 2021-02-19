#pragma once

#ifndef _MSC_VER
extern "C" {
#include <csim/type.h>
}
#else
#include <csim/type.h>
#endif

#include <Eigen/Core>
#include <Eigen/Sparse>
#include <cereal/access.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/complex.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/tuple.hpp>
#include <cereal/types/vector.hpp>
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
template <class Archive>
void save(Archive& ar, const ComplexMatrix& m) {
    int32_t rows = m.rows();
    int32_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void load(Archive& ar, ComplexMatrix& m) {
    int32_t rows;
    int32_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void save(Archive& ar, const ComplexVector& m) {
    int32_t rows = m.rows();
    int32_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void load(Archive& ar, ComplexVector& m) {
    int32_t rows;
    int32_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void save(Archive& ar, const SparseComplexMatrix& m) {
    int32_t rows = m.rows();
    int32_t cols = m.cols();
    ar(rows);
    ar(cols);
    std::vector<std::tuple<int, int, CPPCTYPE>> TripletList;
    for (int k = 0; k < m.outerSize(); ++k) {
        for (Eigen::SparseMatrix<CPPCTYPE>::InnerIterator it(m, k); it; ++it) {
            TripletList.push_back(
                std::tuple<int, int, CPPCTYPE>(it.row(), it.col(), it.value()));
        }
    }
    ar(TripletList);
}

template <class Archive>
void load(Archive& ar, SparseComplexMatrix& m) {
    int32_t rows;
    int32_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    std::vector<std::tuple<int, int, CPPCTYPE>> TripletList;
    ar(TripletList);
    std::vector<Eigen::Triplet<CPPCTYPE>> Triplets;
    for (UINT i = 0; i < TripletList.size(); ++i) {
        Triplets.push_back(Eigen::Triplet<CPPCTYPE>(std::get<0>(TripletList[i]),
            std::get<1>(TripletList[i]), std::get<2>(TripletList[i])));
    }
    m.setFromTriplets(Triplets.begin(), Triplets.end());
}

template <class Archive>
void save(Archive& ar, const Eigen::Triplet<CPPCTYPE>& m) {
    ar(m.row());
    ar(m.col());
    ar(m.value());
}

template <class Archive>
void load(Archive& ar, Eigen::Triplet<CPPCTYPE>& m) {
    int row, col, value;
    ar(row, col, value);
    m = Eigen::Triplet<CPPCTYPE>(row, col, value);
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
