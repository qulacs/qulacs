
#include <gtest/gtest.h>
#include "../util/util.h"
#include <Eigen/Core>
#include <string>
#include <algorithm>

#ifndef _MSC_VER
extern "C" {
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
}
#else
#include <csim/memory_ops.h>
#include <csim/stat_ops.h>
#include <csim/update_ops.h>
#include <csim/init_ops.h>
#endif
#include <csim/update_ops_cpp.hpp>

void test_single_control_single_target(std::function<void(UINT,UINT,UINT,const CTYPE*,CTYPE*,ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target,control;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit control-1 single qubit gate
        target = rand_int(n);
        control = rand_int(n - 1);
        if (control >= target) control++;
        U = get_eigen_matrix_random_single_qubit_unitary();
        func(control, 1, target, (CTYPE*)U.data(), state, dim);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n)*get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
        state_equal(state, test_state, dim, "single qubit control sinlge qubit dense gate");

        // single qubit control-0 single qubit gate
        target = rand_int(n);
        control = rand_int(n - 1);
        if (control >= target) control++;
        U = get_eigen_matrix_random_single_qubit_unitary();
        func(control, 0, target, (CTYPE*)U.data(), state, dim);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P1, n) + get_expanded_eigen_matrix_with_identity(control, P0, n)*get_expanded_eigen_matrix_with_identity(target, U, n)) * test_state;
        state_equal(state, test_state, dim, "single qubit control sinlge qubit dense gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleQubitControlSingleQubitDenseMatrixTest) {
	test_single_control_single_target(single_qubit_control_single_qubit_dense_matrix_gate);
	test_single_control_single_target(single_qubit_control_single_qubit_dense_matrix_gate_single_unroll);
#ifdef _OPENMP
	test_single_control_single_target(single_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll);
#endif
#ifdef _USE_SIMD
	test_single_control_single_target(single_qubit_control_single_qubit_dense_matrix_gate_single_simd);
#ifdef _OPENMP
	test_single_control_single_target(single_qubit_control_single_qubit_dense_matrix_gate_parallel_simd);
#endif
#endif
}

void test_two_control_single_target(std::function<void(const UINT*, const UINT*, UINT, UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U;

    UINT target;
    UINT controls[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // two qubit control-10 single qubit gate
        std::random_shuffle(index_list.begin(), index_list.end());
        target = index_list[0];
        controls[0] = index_list[1];
        controls[1] = index_list[2];

        U = get_eigen_matrix_random_single_qubit_unitary();
        UINT mvalues[2] = { 1,0 };
        func(controls, mvalues, 2, target, (CTYPE*)U.data(), state, dim);
        test_state = (
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n)*get_expanded_eigen_matrix_with_identity(target, U, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n)
            ) * test_state;
        state_equal(state, test_state, dim, "two qubit control sinlge qubit dense gate");

    }
    release_quantum_state(state);
}

TEST(UpdateTest, TwoQubitControlSingleQubitDenseMatrixTest) {
	test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate);
	test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate_single_unroll);
#ifdef _OPENMP
	test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate_parallel_unroll);
#endif
#ifdef _USE_SIMD
	test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate_single_simd);
#ifdef _OPENMP
	test_two_control_single_target(multi_qubit_control_single_qubit_dense_matrix_gate_parallel_simd);
#endif
#endif
}


TEST(UpdateTest, SingleQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], control;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // single qubit 1-controlled qubit dense matrix gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();
        std::random_shuffle(index_list.begin(), index_list.end());
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        control = index_list[2];

        Umerge = kronecker_product(U2, U);
        test_state = (get_expanded_eigen_matrix_with_identity(control, P0, n) + get_expanded_eigen_matrix_with_identity(control, P1, n)*get_expanded_eigen_matrix_with_identity(targets[1], U2, n) * get_expanded_eigen_matrix_with_identity(targets[0], U, n)) * test_state;
        single_qubit_control_multi_qubit_dense_matrix_gate(control, 1, targets, 2, (CTYPE*)Umerge.data(), state, dim);
        state_equal(state, test_state, dim, "single qubit control two-qubit separable dense gate");
    }
    release_quantum_state(state);
}



TEST(UpdateTest, TwoQubitControlTwoQubitDenseMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::MatrixXcd P0(2, 2), P1(2, 2);
    P0 << 1, 0, 0, 0;
    P1 << 0, 0, 0, 1;

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2], controls[2],mvalues[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i) test_state[i] = state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    for (UINT rep = 0; rep < max_repeat; ++rep) {

        // two qubit control-11 two qubit gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();
        std::random_shuffle(index_list.begin(), index_list.end());
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        controls[0] = index_list[2];
        controls[1] = index_list[3];

        mvalues[0] = 1; mvalues[1] = 1;
        Umerge = kronecker_product(U2, U);
        multi_qubit_control_multi_qubit_dense_matrix_gate(controls, mvalues, 2, targets, 2, (CTYPE*)Umerge.data(), state, dim);
        test_state = (
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P0, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P0, n) +
            get_expanded_eigen_matrix_with_identity(controls[0], P1, n)*get_expanded_eigen_matrix_with_identity(controls[1], P1, n)*get_expanded_eigen_matrix_with_identity(targets[0], U, n)*get_expanded_eigen_matrix_with_identity(targets[1], U2, n)
            ) * test_state;
        state_equal(state, test_state, dim, "two qubit control two qubit dense gate");
    }
    release_quantum_state(state);
}
