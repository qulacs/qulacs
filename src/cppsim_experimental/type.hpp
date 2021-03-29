#pragma once

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
#include <csim/type.hpp>
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
    size_t rows = m.rows();
    size_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void load(Archive& ar, ComplexMatrix& m) {
    size_t rows;
    size_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void save(Archive& ar, const ComplexVector& m) {
    size_t rows = m.rows();
    size_t cols = m.cols();
    ar(rows);
    ar(cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void load(Archive& ar, ComplexVector& m) {
    size_t rows;
    size_t cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    ar(binary_data(
        m.data(), static_cast<std::size_t>(rows * cols * sizeof(CPPCTYPE))));
}

template <class Archive>
void save(Archive& ar, const SparseComplexMatrix& m) {
    Eigen::Index rows = m.rows();
    Eigen::Index cols = m.cols();
    ar(rows);
    ar(cols);
    std::vector<std::tuple<Eigen::Index, Eigen::Index, CPPCTYPE>> TripletList;
    for (Eigen::Index k = 0; k < m.outerSize(); ++k) {
        for (Eigen::SparseMatrix<CPPCTYPE>::InnerIterator it(m, k); it; ++it) {
            TripletList.push_back(
                std::tuple<Eigen::Index, Eigen::Index, CPPCTYPE>(
                    it.row(), it.col(), it.value()));
        }
    }
    ar(TripletList);
}

template <class Archive>
void load(Archive& ar, SparseComplexMatrix& m) {
    Eigen::Index rows;
    Eigen::Index cols;
    ar(rows);
    ar(cols);

    m.resize(rows, cols);
    std::vector<std::tuple<Eigen::Index, Eigen::Index, CPPCTYPE>> TripletList;
    ar(TripletList);
    std::vector<Eigen::Triplet<CPPCTYPE>> Triplets;
    for (size_t i = 0; i < TripletList.size(); ++i) {
        Triplets.push_back(
            Eigen::Triplet<CPPCTYPE>((int)std::get<0>(TripletList[i]),
                (int)std::get<1>(TripletList[i]), std::get<2>(TripletList[i])));
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
