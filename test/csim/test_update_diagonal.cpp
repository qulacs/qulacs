
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <csim/init_ops.hpp>
#include <csim/memory_ops.hpp>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_cpp.hpp>
#include <string>

#include "../util/util.hpp"

void test_single_diagonal_matrix_gate(
    std::function<void(UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::complex<double> imag_unit(0, 1);
    const auto Identity = make_Identity();
    const auto Z = make_Z();

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit diagonal matrix gate
        const auto target = rand_int(n);
        auto icoef = rand_real();
        auto zcoef = rand_real();
        const auto norm = sqrt(icoef * icoef + zcoef * zcoef);
        icoef /= norm;
        zcoef /= norm;
        U = icoef * Identity + imag_unit * zcoef * Z;
        Eigen::VectorXcd diag = U.diagonal();
        func(target, (CTYPE*)diag.data(), state, dim);
        test_state =
            get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
        state_equal(state, test_state, dim, "single diagonal gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleDiagonalMatrixTest) {
    test_single_diagonal_matrix_gate(single_qubit_diagonal_matrix_gate);
    test_single_diagonal_matrix_gate(
        single_qubit_diagonal_matrix_gate_parallel_unroll);
#ifdef _USE_SIMD
    test_single_diagonal_matrix_gate(
        single_qubit_diagonal_matrix_gate_parallel_simd);
#endif
#ifdef _USE_SVE
    test_single_diagonal_matrix_gate(
        single_qubit_diagonal_matrix_gate_parallel_sve);
#endif
}

void test_single_phase_gate(
    std::function<void(UINT, CTYPE, CTYPE*, ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit phase matrix gate
        target = rand_int(n);
        angle = rand_real();
        U << 1, 0, 0, cos(angle) + 1.i * sin(angle);
        CTYPE t = cos(angle) + 1.i * sin(angle);
        func(target, t, state, dim);
        test_state =
            get_expanded_eigen_matrix_with_identity(target, U, n) * test_state;
        state_equal(state, test_state, dim, "single phase gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SinglePhaseGateTest) {
    test_single_phase_gate(single_qubit_phase_gate);
    test_single_phase_gate(single_qubit_phase_gate_parallel_unroll);
#ifdef _USE_SIMD
    test_single_phase_gate(single_qubit_phase_gate_parallel_simd);
#endif
}
