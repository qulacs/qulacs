#pragma once

#include <gtest/gtest.h>

#include <Eigen/Core>
#include <array>
#include <cmath>
#include <csim/type.hpp>
#include <string>

#include "cppsim/state.hpp"

#ifdef __GNUC__
#if __GNUC__ >= 8
using namespace std::complex_literals;
#endif
#endif

const double eps = 1e-12;
// random
static UINT rand_int(UINT max) { return rand() % max; }

static double rand_real() { return (rand() % RAND_MAX) / ((double)RAND_MAX); }

// obtain single dense matrix
static Eigen::MatrixXcd get_eigen_matrix_single_Pauli(UINT pauli_id) {
    Eigen::MatrixXcd mat(2, 2);
    if (pauli_id == 0)
        mat << 1, 0, 0, 1;
    else if (pauli_id == 1)
        mat << 0, 1, 1, 0;
    else if (pauli_id == 2)
        mat << 0, -1.i, 1.i, 0;
    else if (pauli_id == 3)
        mat << 1, 0, 0, -1;
    return mat;
}

static Eigen::MatrixXcd get_eigen_matrix_random_single_qubit_unitary() {
    Eigen::MatrixXcd Identity, X, Y, Z;
    Identity = get_eigen_matrix_single_Pauli(0);
    X = get_eigen_matrix_single_Pauli(1);
    Y = get_eigen_matrix_single_Pauli(2);
    Z = get_eigen_matrix_single_Pauli(3);

    double icoef, xcoef, ycoef, zcoef, norm;
    icoef = rand_real();
    xcoef = rand_real();
    ycoef = rand_real();
    zcoef = rand_real();
    norm = sqrt(icoef * icoef + xcoef + xcoef + ycoef * ycoef + zcoef * zcoef);
    icoef /= norm;
    xcoef /= norm;
    ycoef /= norm;
    zcoef /= norm;
    return icoef * Identity + 1.i * xcoef * X + 1.i * ycoef * Y +
           1.i * zcoef * Z;
}

static Eigen::VectorXcd get_eigen_diagonal_matrix_random_multi_qubit_unitary(
    UINT qubit_count) {
    ITYPE dim = (1ULL) << qubit_count;
    auto vec = Eigen::VectorXcd(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        double angle = rand_real() * 2 * 3.14159;
        vec[i] = cos(angle) + 1.i * sin(angle);
    }
    return vec;
}

// expand matrix
static Eigen::MatrixXcd kronecker_product(
    const Eigen::MatrixXcd& lhs, const Eigen::MatrixXcd& rhs) {
    Eigen::MatrixXcd result(lhs.rows() * rhs.rows(), lhs.cols() * rhs.cols());
    for (int i = 0; i < lhs.rows(); i++) {
        for (int j = 0; j < lhs.cols(); j++) {
            result.block(i * rhs.rows(), j * rhs.cols(), rhs.rows(),
                rhs.cols()) = lhs(i, j) * rhs;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_expanded_eigen_matrix_with_identity(
    UINT target_qubit_index, const Eigen::MatrixXcd& one_qubit_matrix,
    UINT qubit_count) {
    const ITYPE left_dim = 1ULL << target_qubit_index;
    const ITYPE right_dim = 1ULL << (qubit_count - target_qubit_index - 1);
    auto left_identity = Eigen::MatrixXcd::Identity(left_dim, left_dim);
    auto right_identity = Eigen::MatrixXcd::Identity(right_dim, right_dim);
    return kronecker_product(
        kronecker_product(right_identity, one_qubit_matrix), left_identity);
}

// get expanded matrix

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(
    std::vector<UINT> pauli_ids) {
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Identity(1, 1);
    for (UINT i = 0; i < pauli_ids.size(); ++i) {
        result = kronecker_product(
            get_eigen_matrix_single_Pauli(pauli_ids[i]), result)
                     .eval();
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_pauli(
    std::vector<UINT> index_list, std::vector<UINT> pauli_list,
    UINT qubit_count) {
    std::vector<UINT> whole_pauli_ids(qubit_count, 0);
    for (UINT i = 0; i < index_list.size(); ++i) {
        whole_pauli_ids[index_list[i]] = pauli_list[i];
    }
    return get_eigen_matrix_full_qubit_pauli(whole_pauli_ids);
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CNOT(
    UINT control_qubit_index, UINT target_qubit_index, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        if (ind & (1ULL << control_qubit_index)) {
            result(ind, ind ^ (1ULL << target_qubit_index)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_CZ(
    UINT control_qubit_index, UINT target_qubit_index, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        if ((ind & (1ULL << control_qubit_index)) != 0 &&
            (ind & (1ULL << target_qubit_index)) != 0) {
            result(ind, ind) = -1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

static Eigen::MatrixXcd get_eigen_matrix_full_qubit_SWAP(
    UINT target_qubit_index1, UINT target_qubit_index2, UINT qubit_count) {
    ITYPE dim = 1ULL << qubit_count;
    Eigen::MatrixXcd result = Eigen::MatrixXcd::Zero(dim, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        bool flag1, flag2;
        flag1 = (ind & (1ULL << target_qubit_index1)) != 0;
        flag2 = (ind & (1ULL << target_qubit_index2)) != 0;
        if (flag1 ^ flag2) {
            result(ind, ind ^ (1ULL << target_qubit_index1) ^
                            (1ULL << target_qubit_index2)) = 1;
        } else {
            result(ind, ind) = 1;
        }
    }
    return result;
}

// utils
static std::string convert_CTYPE_array_to_string(
    const CTYPE* state, ITYPE dim) {
    std::string str = "";
    for (ITYPE ind = 0; ind < dim; ++ind) {
        str += "(" + std::to_string(_creal(state[ind])) + "," +
               std::to_string(_cimag(state[ind])) + ") ";
    }
    return str;
}

static Eigen::VectorXcd convert_CTYPE_array_to_eigen_vector(
    const CTYPE* state, ITYPE dim) {
    Eigen::VectorXcd vec(dim);
    for (ITYPE i = 0; i < dim; ++i) vec[i] = state[i];
    return vec;
}

static void state_equal(const CTYPE* state, const Eigen::VectorXcd& test_state,
    ITYPE dim, std::string gate_string) {
    Eigen::VectorXcd vec = convert_CTYPE_array_to_eigen_vector(state, dim);
    for (ITYPE ind = 0; ind < dim; ++ind) {
        ASSERT_NEAR(abs(vec[ind] - test_state[ind]), 0, eps)
            << gate_string << " at " << ind << std::endl
            << "Eigen : " << test_state.transpose() << std::endl
            << "CSIM : " << convert_CTYPE_array_to_string(state, dim)
            << std::endl;
    }
}

#define ASSERT_STATE_NEAR(state, other, eps) \
    ASSERT_PRED_FORMAT3(_assert_state_near, state, other, eps)

#define EXPECT_STATE_NEAR(state, other, eps) \
    EXPECT_PRED_FORMAT3(_assert_state_near, state, other, eps)

static testing::AssertionResult _assert_state_near(const char* state1_name,
    const char* state2_name, const char* eps_name, const QuantumState& state1,
    const QuantumState& state2, const double eps) {
    if (state1.dim != state2.dim) {
        return testing::AssertionFailure()
               << "The dimension is different\nDimension of " << state1_name
               << " is " << state1.dim << ",\n"
               << "Dimension of " << state2_name << " is " << state2.dim << ".";
    }

    for (UINT i = 0; i < state1.dim; i++) {
        const double real_diff = std::fabs(
            state1.data_cpp()[i].real() - state2.data_cpp()[i].real());
        if (real_diff > eps) {
            return testing::AssertionFailure()
                   << "The difference between " << i << "-th real part of "
                   << state1_name << " and " << state2_name << " is "
                   << real_diff << ", which exceeds " << eps << ", where\n"
                   << state1_name << " evaluates to "
                   << state1.data_cpp()[i].real() << ",\n"
                   << state2_name << " evaluates to "
                   << state2.data_cpp()[i].real() << ", and\n"
                   << eps_name << " evaluates to " << eps << ".";
        }

        const double imag_diff = std::fabs(
            state1.data_cpp()[i].real() - state2.data_cpp()[i].real());
        if (imag_diff > eps) {
            return testing::AssertionFailure()
                   << "The difference between " << i << "-th imaginary part of "
                   << state1_name << " and " << state2_name << " is "
                   << imag_diff << ", which exceeds " << eps << ", where\n"
                   << state1_name << " evaluates to "
                   << state1.data_cpp()[i].imag() << ",\n"
                   << state2_name << " evaluates to "
                   << state2.data_cpp()[i].imag() << ", and\n"
                   << eps_name << " evaluates to " << eps << ".";
        }
    }

    return testing::AssertionSuccess();
}

#define _CHECK_NEAR(val1, val2, eps) \
    _check_near(val1, val2, eps, #val1, #val2, #eps, __FILE__, __LINE__)
static std::string _check_near(double val1, double val2, double eps,
    std::string val1_name, std::string val2_name, std::string eps_name,
    std::string file, UINT line) {
    double diff = std::abs(val1 - val2);
    if (diff <= eps) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "The difference between " << val1_name << " and "
                         << val2_name << " is " << diff << ", which exceeds "
                         << eps_name << ", where\n"
                         << val1_name << " evaluates to " << val1 << ",\n"
                         << val2_name << " evaluates to " << val2 << ", and\n"
                         << eps_name << " evaluates to " << eps << ".\n";
    return error_message_stream.str();
}

#define _CHECK_LT(val1, val2) \
    _check_lt(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
static std::string _check_lt(T val1, T val2, std::string val1_name,
    std::string val2_name, std::string file, UINT line) {
    if (val1 < val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") < (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

#define _CHECK_LE(val1, val2) \
    _check_le(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
static std::string _check_le(T val1, T val2, std::string val1_name,
    std::string val2_name, std::string file, UINT line) {
    if (val1 <= val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") <= (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

#define _CHECK_GT(val1, val2) \
    _check_gt(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
static std::string _check_gt(T val1, T val2, std::string val1_name,
    std::string val2_name, std::string file, UINT line) {
    if (val1 > val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") > (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

#define _CHECK_GE(val1, val2) \
    _check_ge(val1, val2, #val1, #val2, __FILE__, __LINE__)
template <typename T>
static std::string _check_ge(T val1, T val2, std::string val1_name,
    std::string val2_name, std::string file, UINT line) {
    if (val1 >= val2) return "";
    std::stringstream error_message_stream;
    error_message_stream << file << ":" << line << ": Failure\n"
                         << "Expected: (" << val1_name << ") >= (" << val2_name
                         << "), actual: " << val1 << " vs " << val2 << "\n";
    return error_message_stream.str();
}

static Eigen::MatrixXcd make_2x2_matrix(const Eigen::dcomplex a00,
    const Eigen::dcomplex a01, const Eigen::dcomplex a10,
    const Eigen::dcomplex a11) {
    Eigen::MatrixXcd m(2, 2);
    m << a00, a01, a10, a11;
    return m;
}

static Eigen::MatrixXcd make_Identity() {
    return Eigen::MatrixXcd::Identity(2, 2);
}

static Eigen::MatrixXcd make_X() { return make_2x2_matrix(0, 1, 1, 0); }

static Eigen::MatrixXcd make_Y() { return make_2x2_matrix(0, -1.i, 1.i, 0); }

static Eigen::MatrixXcd make_Z() { return make_2x2_matrix(1, 0, 0, -1); }

static Eigen::MatrixXcd make_H() {
    return make_2x2_matrix(
        1 / sqrt(2.), 1 / sqrt(2.), 1 / sqrt(2.), -1 / sqrt(2.));
}

static Eigen::MatrixXcd make_S() { return make_2x2_matrix(1, 0, 0, 1.i); }

static Eigen::MatrixXcd make_T() {
    return make_2x2_matrix(1, 0, 0, (1. + 1.i) / sqrt(2.));
}

static Eigen::MatrixXcd make_sqrtX() {
    return make_2x2_matrix(0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i);
}

static Eigen::MatrixXcd make_sqrtY() {
    return make_2x2_matrix(0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i);
}

static Eigen::MatrixXcd make_P0() { return make_2x2_matrix(1, 0, 0, 0); }

static Eigen::MatrixXcd make_P1() { return make_2x2_matrix(0, 0, 0, 1); }
