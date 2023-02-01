
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

TEST(UpdateTest, SingleQubitPauliTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT target, pauli;

    const auto Identity = make_Identity();

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        /* single qubit Pauli gate */
        target = rand_int(n);
        pauli = rand_int(4);
        single_qubit_Pauli_gate(target, pauli, state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(
                         target, get_eigen_matrix_single_Pauli(pauli), n) *
                     test_state;
        state_equal(state, test_state, dim, "single Pauli gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, SingleQubitPauliRotationTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    const auto Identity = make_Identity();

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];
    std::complex<double> imag_unit(0, 1);

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        const auto target = rand_int(n);
        const auto pauli = rand_int(3) + 1;
        const auto angle = rand_real();
        single_qubit_Pauli_rotation_gate(target, pauli, angle, state, dim);
        test_state = get_expanded_eigen_matrix_with_identity(target,
                         cos(angle / 2) * Identity +
                             imag_unit * sin(angle / 2) *
                                 get_eigen_matrix_single_Pauli(pauli),
                         n) *
                     test_state;
        state_equal(state, test_state, dim, "single rotation Pauli gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, MultiQubitPauliTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        // multi pauli whole
        std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
        std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

        pauli_whole.resize(n);
        for (UINT i = 0; i < n; ++i) {
            pauli_whole[i] = rand_int(4);
        }
        multi_qubit_Pauli_gate_whole_list(pauli_whole.data(), n, state, dim);
        test_state =
            get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
        state_equal(state, test_state, dim, "multi Pauli whole gate");

        // multi pauli partial
        pauli_partial.clear();
        pauli_partial_index.clear();
        pauli_partial_pair.clear();
        for (UINT i = 0; i < n; ++i) {
            const auto pauli = rand_int(4);
            pauli_whole[i] = pauli;
            if (pauli != 0) {
                pauli_partial_pair.push_back(std::make_pair(i, pauli));
            }
        }
        std::shuffle(
            pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
        for (auto val : pauli_partial_pair) {
            pauli_partial_index.push_back(val.first);
            pauli_partial.push_back(val.second);
        }
        multi_qubit_Pauli_gate_partial_list(pauli_partial_index.data(),
            pauli_partial.data(), (UINT)pauli_partial.size(), state, dim);
        test_state =
            get_eigen_matrix_full_qubit_pauli(pauli_whole) * test_state;
        state_equal(state, test_state, dim, "multi Pauli partial gate");
    }
    release_quantum_state(state);
}

TEST(UpdateTest, MultiQubitPauliRotationTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    UINT pauli;
    double angle;

    auto state = allocate_quantum_state(dim);
    initialize_Haar_random_state(state, dim);
    Eigen::VectorXcd test_state = Eigen::VectorXcd::Zero(dim);
    for (ITYPE i = 0; i < dim; ++i)
        test_state[i] = (std::complex<double>)state[i];

    Eigen::MatrixXcd whole_I = Eigen::MatrixXcd::Identity(dim, dim);
    std::complex<double> imag_unit(0, 1);

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        std::vector<UINT> pauli_whole, pauli_partial, pauli_partial_index;
        std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

        // multi pauli rotation whole
        pauli_whole.resize(n);
        for (UINT i = 0; i < n; ++i) {
            pauli_whole[i] = rand_int(4);
        }
        angle = rand_real();
        multi_qubit_Pauli_rotation_gate_whole_list(
            pauli_whole.data(), n, angle, state, dim);
        test_state = (cos(angle / 2) * whole_I +
                         imag_unit * sin(angle / 2) *
                             get_eigen_matrix_full_qubit_pauli(pauli_whole)) *
                     test_state;
        state_equal(state, test_state, dim, "multi Pauli rotation whole gate");

        // multi pauli rotation partial
        pauli_partial.clear();
        pauli_partial_index.clear();
        pauli_partial_pair.clear();
        for (UINT i = 0; i < n; ++i) {
            pauli = rand_int(4);
            pauli_whole[i] = pauli;
            if (pauli != 0) {
                pauli_partial_pair.push_back(std::make_pair(i, pauli));
            }
        }
        std::shuffle(
            pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
        for (auto val : pauli_partial_pair) {
            pauli_partial_index.push_back(val.first);
            pauli_partial.push_back(val.second);
        }
        angle = rand_real();
        multi_qubit_Pauli_rotation_gate_partial_list(pauli_partial_index.data(),
            pauli_partial.data(), (UINT)pauli_partial.size(), angle, state,
            dim);
        test_state = (cos(angle / 2) * whole_I +
                         imag_unit * sin(angle / 2) *
                             get_eigen_matrix_full_qubit_pauli(pauli_whole)) *
                     test_state;
        state_equal(
            state, test_state, dim, "multi Pauli rotation partial gate");
    }
    release_quantum_state(state);
}
