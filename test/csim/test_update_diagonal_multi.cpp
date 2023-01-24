
#include <gtest/gtest.h>

#include <Eigen/Core>
#include <algorithm>
#include <csim/init_ops.hpp>
#include <csim/memory_ops.hpp>
#include <csim/stat_ops.hpp>
#include <csim/update_ops.hpp>
#include <string>

#include "../util/util.hpp"

TEST(UpdateTest, MultiQubitDiagonalMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT gate_size = 1; gate_size <= 1; ++gate_size) {
        ITYPE gate_dim = (1ULL) << gate_size;
        for (UINT r = 0; r < max_repeat; ++r) {
            std::shuffle(index_list.begin(), index_list.end(), engine);
            auto diag =
                get_eigen_diagonal_matrix_random_multi_qubit_unitary(gate_size);
            Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(dim, dim);
            for (ITYPE i = 0; i < dim; ++i) {
                UINT ti = 0;
                for (int j = 0; j < gate_size; ++j) {
                    UINT gi = index_list[j];
                    ti += ((i >> gi) % 2) * (1 << j);
                }
                matrix(i, i) = diag[ti];
            }
            // std::cout << test_state << std::endl;
            // std::cout << diag << std::endl;
            // std::cout << matrix << std::endl;
            multi_qubit_diagonal_matrix_gate(
                index_list.data(), gate_size, (CTYPE*)diag.data(), state, dim);
            test_state = matrix * test_state;
            state_equal(state, test_state, dim, "diagonal gate");
        }
    }
    release_quantum_state(state);
}

TEST(UpdateTest, MultiQubitDiagonalMatrixTest2) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U1, U2, U3;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT gate_size = 1; gate_size <= 1; ++gate_size) {
        ITYPE gate_dim = (1ULL) << gate_size;
        for (UINT r = 0; r < max_repeat; ++r) {
            std::shuffle(index_list.begin(), index_list.end(), engine);
            auto diag =
                get_eigen_diagonal_matrix_random_multi_qubit_unitary(gate_size);
            Eigen::MatrixXcd matrix = Eigen::MatrixXcd::Zero(dim, dim);
            for (ITYPE i = 0; i < dim; ++i) {
                UINT ti = 0;
                for (int j = 0; j < gate_size; ++j) {
                    UINT gi = index_list[j];
                    ti += ((i >> gi) % 2) * (1 << j);
                }
                matrix(i, i) = diag[ti];
            }
            // std::cout << test_state << std::endl;
            // std::cout << diag << std::endl;
            // std::cout << matrix << std::endl;
            multi_qubit_control_multi_qubit_diagonal_matrix_gate({}, {}, 0,
                index_list.data(), gate_size, (CTYPE*)diag.data(), state, dim);
            test_state = matrix * test_state;
            state_equal(state, test_state, dim, "diagonal gate");
        }
    }
    release_quantum_state(state);
}

TEST(UpdateTest, TwoQubitControlTwoQubitDiagonalMatrixTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    const auto P0 = make_P0();
    const auto P1 = make_P1();

    Eigen::VectorXcd d, d2;

    UINT targets[2], controls[2], mvalues[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // two qubit control-11 two qubit gate
        d = get_eigen_diagonal_matrix_random_multi_qubit_unitary(1);
        d2 = get_eigen_diagonal_matrix_random_multi_qubit_unitary(1);
        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = index_list[0];
        targets[1] = index_list[1];
        controls[0] = index_list[2];
        controls[1] = index_list[3];

        mvalues[0] = 1;
        mvalues[1] = 1;
        Eigen::Vector4cd dmerge;
        dmerge(0) = d[0] * d2[0];
        dmerge(1) = d[1] * d2[0];
        dmerge(2) = d[0] * d2[1];
        dmerge(3) = d[1] * d2[1];
        multi_qubit_control_multi_qubit_diagonal_matrix_gate(controls, mvalues,
            2, targets, 2, (CTYPE*)dmerge.data(), state, dim);

        Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
        U(0, 0) = d[0];
        U(1, 1) = d[1];
        U(0, 1) = 0;
        U(1, 0) = 0;
        U2(0, 0) = d2[0];
        U2(1, 1) = d2[1];
        U2(0, 1) = 0;
        U2(1, 0) = 0;

        test_state =
            (get_expanded_eigen_matrix_with_identity(controls[0], P0, n) *
                    get_expanded_eigen_matrix_with_identity(
                        controls[1], P0, n) +
                get_expanded_eigen_matrix_with_identity(controls[0], P0, n) *
                    get_expanded_eigen_matrix_with_identity(
                        controls[1], P1, n) +
                get_expanded_eigen_matrix_with_identity(controls[0], P1, n) *
                    get_expanded_eigen_matrix_with_identity(
                        controls[1], P0, n) +
                get_expanded_eigen_matrix_with_identity(controls[0], P1, n) *
                    get_expanded_eigen_matrix_with_identity(
                        controls[1], P1, n) *
                    get_expanded_eigen_matrix_with_identity(targets[0], U, n) *
                    get_expanded_eigen_matrix_with_identity(
                        targets[1], U2, n)) *
            test_state;
        state_equal(state, test_state, dim,
            "two qubit control two qubit diagonal gate");
    }
    release_quantum_state(state);
}
