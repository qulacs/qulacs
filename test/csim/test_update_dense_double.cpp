
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

void test_double_dense_matrix_gate(
    std::function<void(UINT, UINT, const CTYPE*, CTYPE*, ITYPE)> func) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    std::vector<UINT> index_list;
    for (UINT i = 0; i < n; ++i) index_list.push_back(i);

    Eigen::Matrix<std::complex<double>, 2, 2, Eigen::RowMajor> U, U2;
    Eigen::Matrix<std::complex<double>, 4, 4, Eigen::RowMajor> Umerge;

    UINT targets[2];

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = static_cast<std::complex<double>>(state[i]);

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // two qubit dense matrix gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();

        std::shuffle(index_list.begin(), index_list.end(), engine);

        targets[0] = index_list[0];
        targets[1] = index_list[1];
        Umerge = kronecker_product(U2, U);
        // the below two lines are equivalent to the above two line
        // UINT targets_rev[2] = { targets[1], targets[0] };
        // Umerge = kronecker_product(U, U2);
        test_state =
            get_expanded_eigen_matrix_with_identity(targets[1], U2, n) *
            get_expanded_eigen_matrix_with_identity(targets[0], U, n) *
            test_state;
        func(targets[0], targets[1], (CTYPE*)Umerge.data(), state, dim);
        state_equal(state, test_state, dim, "two-qubit separable dense gate");
    }

    std::vector<std::pair<UINT, UINT>> test_pairs;
    for (UINT i = 0; i < n; ++i) {
        for (UINT j = 0; j < n; ++j) {
            if (i == j) continue;
            test_pairs.push_back(std::make_pair(i, j));
        }
    }

    for (auto pair : test_pairs) {
        // two qubit dense matrix gate
        U = get_eigen_matrix_random_single_qubit_unitary();
        U2 = get_eigen_matrix_random_single_qubit_unitary();

        std::shuffle(index_list.begin(), index_list.end(), engine);
        targets[0] = pair.first;
        targets[1] = pair.second;
        Umerge = kronecker_product(U2, U);
        // the below two lines are equivalent to the above two line
        // UINT targets_rev[2] = { targets[1], targets[0] };
        // Umerge = kronecker_product(U, U2);
        test_state =
            get_expanded_eigen_matrix_with_identity(targets[1], U2, n) *
            get_expanded_eigen_matrix_with_identity(targets[0], U, n) *
            test_state;
        func(targets[0], targets[1], (CTYPE*)Umerge.data(), state, dim);
        state_equal(state, test_state, dim, "two-qubit separable dense gate");
    }

    release_quantum_state(state);
}

TEST(UpdateTest, TwoQubitDenseMatrixTest) {
    test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_c);
    test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_nosimd);
#ifdef _USE_SIMD
    test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_simd);
#endif
#ifdef _USE_SVE
    test_double_dense_matrix_gate(double_qubit_dense_matrix_gate_sve);
#endif
}
