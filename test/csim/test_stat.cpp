#include <gtest/gtest.h>

#include <Eigen/Core>
#include <csim/init_ops.hpp>
#include <csim/memory_ops.hpp>
#include <csim/stat_ops.hpp>

#include "../util/util.hpp"

// post-selection probability check
TEST(StatOperationTest, ProbTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    const auto P0 = make_P0();
    const auto P1 = make_P1();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            double p0 = M0_prob(target, state, dim);
            double p1 = M1_prob(target, state, dim);
            ASSERT_NEAR(
                (get_expanded_eigen_matrix_with_identity(target, P0, n) *
                    test_state)
                    .squaredNorm(),
                p0, eps);
            ASSERT_NEAR(
                (get_expanded_eigen_matrix_with_identity(target, P1, n) *
                    test_state)
                    .squaredNorm(),
                p1, eps);
            ASSERT_NEAR(p0 + p1, 1, eps);
        }
    }
    release_quantum_state(state);
}

// marginal probability check
TEST(StatOperationTest, MarginalProbTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    const auto Identity = make_Identity();
    const auto P0 = make_P0();
    const auto P1 = make_P1();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            // merginal probability check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            std::vector<UINT> index_list, measured_value_list;

            index_list.clear();
            measured_value_list.clear();
            for (UINT i = 0; i < n; ++i) {
                UINT measured_value = rand_int(3);
                if (measured_value != 2) {
                    measured_value_list.push_back(measured_value);
                    index_list.push_back(i);
                }
                if (measured_value == 0) {
                    mat = kronecker_product(P0, mat);
                } else if (measured_value == 1) {
                    mat = kronecker_product(P1, mat);
                } else {
                    mat = kronecker_product(Identity, mat);
                }
            }
            double test_marginal_prob = (mat * test_state).squaredNorm();
            double res =
                marginal_prob(index_list.data(), measured_value_list.data(),
                    (UINT)index_list.size(), state, dim);
            ASSERT_NEAR(test_marginal_prob, res, eps);
        }
    }
    release_quantum_state(state);
}

// entropy
TEST(StatOperationTest, EntropyTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            double ent = 0;
            for (ITYPE ind = 0; ind < dim; ++ind) {
                double prob = norm(test_state[ind]);
                if (prob > eps) ent += -prob * log(prob);
            }
            ASSERT_NEAR(ent, measurement_distribution_entropy(state, dim), eps);
        }
    }
    release_quantum_state(state);
}

// inner product
TEST(StatOperationTest, InnerProductTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    CTYPE* buffer = allocate_quantum_state(dim);
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            initialize_Haar_random_state(buffer, dim);
            CTYPE inp = state_inner_product(buffer, state, dim);
            Eigen::VectorXcd test_buffer(dim);
            for (ITYPE i = 0; i < dim; ++i)
                test_buffer[i] = (std::complex<double>)buffer[i];
            std::complex<double> test_inp =
                (test_buffer.adjoint() * test_state);
            ASSERT_NEAR(_creal(inp), test_inp.real(), eps);
            ASSERT_NEAR(_cimag(inp), test_inp.imag(), eps);
        }
    }
    release_quantum_state(state);
    release_quantum_state(buffer);
}

// single qubit expectation value
TEST(StatOperationTest, SingleQubitExpectationValueTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            // single qubit expectation value check
            target = rand_int(n);
            UINT pauli = rand_int(3) + 1;
            if (pauli == 0)
                pauli_op = Identity;
            else if (pauli == 1)
                pauli_op = X;
            else if (pauli == 2)
                pauli_op = Y;
            else if (pauli == 3)
                pauli_op = Z;
            std::complex<double> value =
                (test_state.adjoint() *
                    get_expanded_eigen_matrix_with_identity(
                        target, pauli_op, n) *
                    test_state);
            ASSERT_NEAR(value.imag(), 0, eps);
            double test_expectation = value.real();
            double expectation = expectation_value_single_qubit_Pauli_operator(
                target, pauli, state, dim);
            ASSERT_NEAR(expectation, test_expectation, eps);
        }
    }
    release_quantum_state(state);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueWholeTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation whole list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            std::vector<UINT> pauli_whole;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(4);
                if (pauli == 0)
                    pauli_op = Identity;
                else if (pauli == 1)
                    pauli_op = X;
                else if (pauli == 2)
                    pauli_op = Y;
                else if (pauli == 3)
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                pauli_whole.push_back(pauli);
            }
            std::complex<double> value =
                (test_state.adjoint() * mat * test_state);
            ASSERT_NEAR(value.imag(), 0, eps);
            double test_expectation = value.real();
            double expectation =
                expectation_value_multi_qubit_Pauli_operator_whole_list(
                    pauli_whole.data(), n, state, dim);
            ASSERT_NEAR(expectation, test_expectation, eps);
        }
    }
    release_quantum_state(state);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueZopWholeTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = static_cast<std::complex<double>>(state[i]);

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation whole list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            std::vector<UINT> pauli_whole;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(2);
                if (pauli == 1) pauli = 3;
                if (pauli == 0)
                    pauli_op = Identity;
                else
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                pauli_whole.push_back(pauli);
            }
            std::complex<double> value =
                (test_state.adjoint() * mat * test_state);
            ASSERT_NEAR(value.imag(), 0, eps);
            double test_expectation = value.real();
            double expectation =
                expectation_value_multi_qubit_Pauli_operator_whole_list(
                    pauli_whole.data(), n, state, dim);
            ASSERT_NEAR(expectation, test_expectation, eps);
        }
    }
    release_quantum_state(state);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValuePartialTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    CTYPE* state = allocate_quantum_state(dim);
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = (std::complex<double>)state[i];

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation partial list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            Eigen::MatrixXcd pauli_op;

            std::vector<UINT> pauli_partial, pauli_index;
            std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(4);
                if (pauli == 0)
                    pauli_op = Identity;
                else if (pauli == 1)
                    pauli_op = X;
                else if (pauli == 2)
                    pauli_op = Y;
                else if (pauli == 3)
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                if (pauli != 0) {
                    pauli_partial_pair.push_back(std::make_pair(i, pauli));
                }
            }
            std::shuffle(
                pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
            for (auto val : pauli_partial_pair) {
                pauli_index.push_back(val.first);
                pauli_partial.push_back(val.second);
            }
            std::complex<double> value =
                (test_state.adjoint() * mat * test_state);
            ASSERT_NEAR(value.imag(), 0, eps);
            double test_expectation = value.real();
            double expectation =
                expectation_value_multi_qubit_Pauli_operator_partial_list(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state, dim);
            ASSERT_NEAR(expectation, test_expectation, eps);
            double expectation_st =
                expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state, dim);
            ASSERT_NEAR(expectation_st, test_expectation, eps);
        }
    }
    release_quantum_state(state);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitExpectationValueZopPartialTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;

    const UINT max_repeat = 10;

    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    CTYPE* state = allocate_quantum_state(dim);
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state, dim);
        ASSERT_NEAR(state_norm_squared(state, dim), 1, eps);
        Eigen::VectorXcd test_state(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state[i] = static_cast<std::complex<double>>(state[i]);

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation partial list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            Eigen::MatrixXcd pauli_op;

            std::vector<UINT> pauli_partial, pauli_index;
            std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(2);
                if (pauli == 1) pauli = 3;
                if (pauli == 0)
                    pauli_op = Identity;
                else
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                if (pauli != 0) {
                    pauli_partial_pair.push_back(std::make_pair(i, pauli));
                }
            }
            std::shuffle(
                pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
            for (auto val : pauli_partial_pair) {
                pauli_index.push_back(val.first);
                pauli_partial.push_back(val.second);
            }
            std::complex<double> value =
                (test_state.adjoint() * mat * test_state);
            ASSERT_NEAR(value.imag(), 0, eps);
            double test_expectation = value.real();
            double expectation =
                expectation_value_multi_qubit_Pauli_operator_partial_list(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state, dim);
            ASSERT_NEAR(expectation, test_expectation, eps);
            double expectation_st =
                expectation_value_multi_qubit_Pauli_operator_partial_list_single_thread(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state, dim);
            ASSERT_NEAR(expectation_st, test_expectation, eps);
        }
    }
    release_quantum_state(state);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudeWholeTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state_ket = allocate_quantum_state(dim);
    CTYPE* state_bra = allocate_quantum_state(dim);
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();
    Eigen::MatrixXcd pauli_op;

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state_ket, dim);
        initialize_Haar_random_state(state_bra, dim);
        ASSERT_NEAR(state_norm_squared(state_ket, dim), 1, eps);
        ASSERT_NEAR(state_norm_squared(state_bra, dim), 1, eps);

        Eigen::VectorXcd test_state_ket(dim);
        Eigen::VectorXcd test_state_bra(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_ket[i] = static_cast<std::complex<double>>(state_ket[i]);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_bra[i] = static_cast<std::complex<double>>(state_bra[i]);

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation whole list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            std::vector<UINT> pauli_whole;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(4);
                if (pauli == 0)
                    pauli_op = Identity;
                else if (pauli == 1)
                    pauli_op = X;
                else if (pauli == 2)
                    pauli_op = Y;
                else if (pauli == 3)
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                pauli_whole.push_back(pauli);
            }
            std::complex<double> test_transition_amplitude =
                (test_state_bra.adjoint() * mat * test_state_ket);
            CTYPE transition_amplitude =
                transition_amplitude_multi_qubit_Pauli_operator_whole_list(
                    pauli_whole.data(), n, state_bra, state_ket, dim);
            ASSERT_NEAR(_creal(transition_amplitude),
                test_transition_amplitude.real(), eps);
            ASSERT_NEAR(_cimag(transition_amplitude),
                test_transition_amplitude.imag(), eps);
        }
    }
    release_quantum_state(state_ket);
    release_quantum_state(state_bra);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudeZopWholeTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state_ket = allocate_quantum_state(dim);
    CTYPE* state_bra = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state_ket, dim);
        initialize_Haar_random_state(state_bra, dim);
        ASSERT_NEAR(state_norm_squared(state_ket, dim), 1, eps);
        ASSERT_NEAR(state_norm_squared(state_bra, dim), 1, eps);

        Eigen::VectorXcd test_state_ket(dim);
        Eigen::VectorXcd test_state_bra(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_ket[i] = (std::complex<double>)state_ket[i];
        for (ITYPE i = 0; i < dim; ++i)
            test_state_bra[i] = (std::complex<double>)state_bra[i];

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation whole list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            std::vector<UINT> pauli_whole;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(2);
                if (pauli == 1) pauli = 3;
                if (pauli == 0)
                    pauli_op = Identity;
                else
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                pauli_whole.push_back(pauli);
            }
            std::complex<double> test_transition_amplitude =
                (test_state_bra.adjoint() * mat * test_state_ket);
            CTYPE transition_amplitude =
                transition_amplitude_multi_qubit_Pauli_operator_whole_list(
                    pauli_whole.data(), n, state_bra, state_ket, dim);
            ASSERT_NEAR(_creal(transition_amplitude),
                test_transition_amplitude.real(), eps);
            ASSERT_NEAR(_cimag(transition_amplitude),
                test_transition_amplitude.imag(), eps);
        }
    }
    release_quantum_state(state_ket);
    release_quantum_state(state_bra);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudePartialTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state_ket = allocate_quantum_state(dim);
    CTYPE* state_bra = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state_ket, dim);
        initialize_Haar_random_state(state_bra, dim);
        ASSERT_NEAR(state_norm_squared(state_ket, dim), 1, eps);
        ASSERT_NEAR(state_norm_squared(state_bra, dim), 1, eps);

        Eigen::VectorXcd test_state_ket(dim);
        Eigen::VectorXcd test_state_bra(dim);
        for (ITYPE i = 0; i < dim; ++i)
            test_state_ket[i] = (std::complex<double>)state_ket[i];
        for (ITYPE i = 0; i < dim; ++i)
            test_state_bra[i] = (std::complex<double>)state_bra[i];

        for (UINT target = 0; target < n; ++target) {
            // multi qubit expectation partial list value check
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            Eigen::MatrixXcd pauli_op;

            std::vector<UINT> pauli_partial, pauli_index;
            std::vector<std::pair<UINT, UINT>> pauli_partial_pair;
            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(4);
                if (pauli == 0)
                    pauli_op = Identity;
                else if (pauli == 1)
                    pauli_op = X;
                else if (pauli == 2)
                    pauli_op = Y;
                else if (pauli == 3)
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                if (pauli != 0) {
                    pauli_partial_pair.push_back(std::make_pair(i, pauli));
                }
            }
            std::shuffle(
                pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
            for (auto val : pauli_partial_pair) {
                pauli_index.push_back(val.first);
                pauli_partial.push_back(val.second);
            }
            std::complex<double> test_transition_amplitude =
                (test_state_bra.adjoint() * mat * test_state_ket);
            CTYPE transition_amplitude =
                transition_amplitude_multi_qubit_Pauli_operator_partial_list(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state_bra, state_ket, dim);
            ASSERT_NEAR(_creal(transition_amplitude),
                test_transition_amplitude.real(), eps);
            ASSERT_NEAR(_cimag(transition_amplitude),
                test_transition_amplitude.imag(), eps);
        }
    }
    release_quantum_state(state_ket);
    release_quantum_state(state_bra);
}

// multi qubit expectation value whole
TEST(StatOperationTest, MultiQubitTransitionAmplitudeZopPartialTest) {
    const UINT n = 6;
    const ITYPE dim = 1ULL << n;
    const UINT max_repeat = 10;

    CTYPE* state_ket = allocate_quantum_state(dim);
    CTYPE* state_bra = allocate_quantum_state(dim);
    Eigen::MatrixXcd pauli_op;
    const auto Identity = make_Identity();
    const auto X = make_X();
    const auto Y = make_Y();
    const auto Z = make_Z();

    std::random_device seed_gen;
    std::mt19937 engine(seed_gen());
    for (UINT rep = 0; rep < max_repeat; ++rep) {
        initialize_Haar_random_state(state_ket, dim);
        initialize_Haar_random_state(state_bra, dim);
        ASSERT_NEAR(state_norm_squared(state_ket, dim), 1, eps);
        ASSERT_NEAR(state_norm_squared(state_bra, dim), 1, eps);

        Eigen::VectorXcd test_state_ket(dim);
        Eigen::VectorXcd test_state_bra(dim);
        for (ITYPE i = 0; i < dim; ++i) {
            test_state_ket[i] = (std::complex<double>)state_ket[i];
        }
        for (ITYPE i = 0; i < dim; ++i) {
            test_state_bra[i] = (std::complex<double>)state_bra[i];
        }

        for (UINT target = 0; target < n; target++) {
            Eigen::MatrixXcd mat = Eigen::MatrixXcd::Identity(1, 1);
            Eigen::MatrixXcd pauli_op;
            std::vector<UINT> pauli_partial, pauli_index;
            std::vector<std::pair<UINT, UINT>> pauli_partial_pair;

            for (UINT i = 0; i < n; ++i) {
                UINT pauli = rand_int(2);
                if (pauli == 1) pauli = 3;
                if (pauli == 0)
                    pauli_op = Identity;
                else
                    pauli_op = Z;
                mat = kronecker_product(pauli_op, mat);
                if (pauli != 0) {
                    pauli_partial_pair.push_back(std::make_pair(i, pauli));
                }
            }
            std::shuffle(
                pauli_partial_pair.begin(), pauli_partial_pair.end(), engine);
            for (auto val : pauli_partial_pair) {
                pauli_index.push_back(val.first);
                pauli_partial.push_back(val.second);
            }
            std::complex<double> test_transition_amplitude =
                (test_state_bra.adjoint() * mat * test_state_ket);
            CTYPE transition_amplitude =
                transition_amplitude_multi_qubit_Pauli_operator_partial_list(
                    pauli_index.data(), pauli_partial.data(),
                    (UINT)pauli_index.size(), state_bra, state_ket, dim);
            ASSERT_NEAR(transition_amplitude.real(),
                test_transition_amplitude.real(), eps);
            ASSERT_NEAR(transition_amplitude.imag(),
                test_transition_amplitude.imag(), eps);
        }
    }
    release_quantum_state(state_ket);
    release_quantum_state(state_bra);
}
