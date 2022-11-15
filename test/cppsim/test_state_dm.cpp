#include <gtest/gtest.h>

#include <cppsim/exception.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>

#include "../util/util.hpp"

TEST(DensityMatrixTest, GenerateAndRelease) {
    UINT n = 5;

    const ITYPE dim = 1ULL << n;
    DensityMatrix state(n);
    ASSERT_EQ(state.qubit_count, n);
    ASSERT_EQ(state.dim, dim);
    state.set_zero_state();
    for (UINT i = 0; i < state.dim; ++i) {
        for (UINT j = 0; j < state.dim; ++j) {
            if (i == 0 && j == 0)
                ASSERT_NEAR(abs(state.data_cpp()[i * dim + j] - 1.), 0, eps);
            else
                ASSERT_NEAR(abs(state.data_cpp()[i * dim + j]), 0, eps);
        }
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64() % state.dim;
        state.set_computational_basis(basis);
        for (UINT i = 0; i < state.dim; ++i) {
            for (UINT j = 0; j < state.dim; ++j) {
                if (i == basis && j == basis)
                    ASSERT_NEAR(
                        abs(state.data_cpp()[i * dim + j] - 1.), 0, eps);
                else
                    ASSERT_NEAR(abs(state.data_cpp()[i * dim + j]), 0, eps);
            }
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state.set_Haar_random_state();
        ASSERT_NEAR(state.get_squared_norm(), 1., eps);
    }
}

TEST(DensityMatrixTest, Sampling) {
    UINT n = 5;
    DensityMatrix state(n);
    state.set_Haar_random_state();
    state.set_computational_basis(10);
    auto res1 = state.sampling(1024);
    state.set_computational_basis(10);
    auto res2 = state.sampling(1024);
}
TEST(DensityMatrixTest, Probabilistic) {
    DensityMatrix state_noI(2);
    DensityMatrix state_yesI(2);
    auto x0 = gate::X(0);
    auto h1 = gate::H(1);
    auto i0 = gate::Identity(0);
    auto proba_gate_noI = QuantumGate_Probabilistic({0.2, 0.2}, {x0, h1});
    auto proba_gate_yesI =
        QuantumGate_Probabilistic({0.2, 0.2, 0.6}, {x0, h1, i0});
    delete x0;
    delete h1;
    delete i0;

    proba_gate_noI.update_quantum_state(&state_noI);
    proba_gate_yesI.update_quantum_state(&state_yesI);

    Observable observable(2);
    observable.add_operator(1.0, "Z 0 Z 1");

    auto res_noI = observable.get_expectation_value(&state_noI);
    auto res_yesI = observable.get_expectation_value(&state_yesI);

    ASSERT_NEAR(res_noI.real(), res_yesI.real(), eps);
}
TEST(DensityMatrixTest, SetState) {
    const UINT n = 5;
    DensityMatrix state(n);
    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector(dim * dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            const auto d = static_cast<double>(i * dim + j);
            state_vector[j * dim + i] =
                d + std::complex<double>(0, 1) * (d + 0.1);
        }
    }
    state.load(state_vector);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            ASSERT_NEAR(state.data_cpp()[i * dim + j].real(),
                state_vector[i * dim + j].real(), eps);
            ASSERT_NEAR(state.data_cpp()[i * dim + j].imag(),
                state_vector[i * dim + j].imag(), eps);
        }
    }
}

TEST(DensityMatrixTest, GetMarginalProbability) {
    const UINT n = 2;
    const ITYPE dim = 1 << n;
    DensityMatrix state(n);
    state.set_Haar_random_state();
    std::vector<double> probs;
    for (ITYPE i = 0; i < dim; ++i) {
        probs.push_back(real(state.data_cpp()[i * dim + i]));
    }
    ASSERT_NEAR(state.get_marginal_probability({0, 0}), probs[0], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 0}), probs[1], eps);
    ASSERT_NEAR(state.get_marginal_probability({0, 1}), probs[2], eps);
    ASSERT_NEAR(state.get_marginal_probability({1, 1}), probs[3], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({0, 2}), probs[0] + probs[2], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({1, 2}), probs[1] + probs[3], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({2, 0}), probs[0] + probs[1], eps);
    ASSERT_NEAR(
        state.get_marginal_probability({2, 1}), probs[2] + probs[3], eps);
    ASSERT_NEAR(state.get_marginal_probability({2, 2}), 1., eps);
}

TEST(DensityMatrixTest, AddState) {
    const UINT n = 5;
    DensityMatrix state1(n);
    DensityMatrix state2(n);
    state1.set_Haar_random_state();
    state2.set_Haar_random_state();

    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector1(dim * dim);
    std::vector<std::complex<double>> state_vector2(dim * dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            state_vector1[i * dim + j] = state1.data_cpp()[i * dim + j];
            state_vector2[i * dim + j] = state2.data_cpp()[i * dim + j];
        }
    }

    state1.add_state(&state2);

    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            ASSERT_NEAR(state1.data_cpp()[i * dim + j].real(),
                state_vector1[i * dim + j].real() +
                    state_vector2[i * dim + j].real(),
                eps);
            ASSERT_NEAR(state1.data_cpp()[i * dim + j].imag(),
                state_vector1[i * dim + j].imag() +
                    state_vector2[i * dim + j].imag(),
                eps);
            ASSERT_NEAR(state2.data_cpp()[i * dim + j].real(),
                state_vector2[i * dim + j].real(), eps);
            ASSERT_NEAR(state2.data_cpp()[i * dim + j].imag(),
                state_vector2[i * dim + j].imag(), eps);
        }
    }
}

TEST(DensityMatrixTest, MultiplyCoef) {
    const UINT n = 10;
    const std::complex<double> coef(0.5, 0.2);

    DensityMatrix state(n);
    state.set_Haar_random_state();

    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector(dim * dim);
    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            state_vector[i * dim + j] = state.data_cpp()[i * dim + j] * coef;
        }
    }
    state.multiply_coef(coef);

    for (ITYPE i = 0; i < dim; ++i) {
        for (ITYPE j = 0; j < dim; ++j) {
            ASSERT_NEAR(state.data_cpp()[i * dim + j].real(),
                state_vector[i * dim + j].real(), eps);
            ASSERT_NEAR(state.data_cpp()[i * dim + j].imag(),
                state_vector[i * dim + j].imag(), eps);
        }
    }
}

TEST(DensityMatrixTest, TensorProduct) {
    const UINT n = 4;

    DensityMatrix state1(n), state2(n);
    state1.set_Haar_random_state();
    state2.set_Haar_random_state();

    DensityMatrix* state3 = state::tensor_product(&state1, &state2);
    // numerical test is performed in python
    delete state3;
}

TEST(DensityMatrixTest, PermutateQubit) {
    const UINT n = 3;

    DensityMatrix state(n);
    state.set_Haar_random_state();
    DensityMatrix* state2 = state::permutate_qubit(&state, {1, 0, 2});
    // numerical test is performed in python
    delete state2;
}

TEST(DensityMatrixTest, PartialTraceSVtoDM) {
    const UINT n = 5;

    DensityMatrix state(n);
    state.set_Haar_random_state();
    DensityMatrix* state2 = state::partial_trace(&state, {2, 0});
    // numerical test is performed in python
    delete state2;
}

TEST(DensityMatrixTest, PartialTraceDMtoDM) {
    const UINT n = 5;

    QuantumState state(n);
    state.set_Haar_random_state();
    DensityMatrix* state2 = state::partial_trace(&state, {2, 0});
    // numerical test is performed in python
    delete state2;
}

TEST(DensityMatrixTest, MakeMixture) {
    UINT n = 4;
    Random random;
    {
        QuantumState state1(n);
        state1.set_Haar_random_state();
        QuantumState state2(n);
        state2.set_Haar_random_state();
        CPPCTYPE coef1(random.uniform(), random.uniform());
        CPPCTYPE coef2(random.uniform(), random.uniform());
        DensityMatrixCpu* mixture =
            state::make_mixture(coef1, &state1, coef2, &state2);
        for (UINT i = 0; i < (1 << n); ++i) {
            for (UINT j = 0; j < (1 << n); ++j) {
                CPPCTYPE val1 =
                    state1.data_cpp()[i] * conj(state1.data_cpp()[j]);
                CPPCTYPE val2 =
                    state2.data_cpp()[i] * conj(state2.data_cpp()[j]);
                ASSERT_NEAR(abs(coef1 * val1 + coef2 * val2 -
                                mixture->data_cpp()[i * (1 << n) + j]),
                    0, eps);
            }
        }
        delete mixture;
    }
    {
        QuantumState state1(n);
        state1.set_Haar_random_state();
        DensityMatrixCpu state2(n);
        state2.set_Haar_random_state();
        CPPCTYPE coef1(random.uniform(), random.uniform());
        CPPCTYPE coef2(random.uniform(), random.uniform());
        DensityMatrixCpu* mixture =
            state::make_mixture(coef1, &state1, coef2, &state2);
        for (UINT i = 0; i < (1 << n); ++i) {
            for (UINT j = 0; j < (1 << n); ++j) {
                CPPCTYPE val1 =
                    state1.data_cpp()[i] * conj(state1.data_cpp()[j]);
                CPPCTYPE val2 = state2.data_cpp()[i * (1 << n) + j];
                ASSERT_NEAR(abs(coef1 * val1 + coef2 * val2 -
                                mixture->data_cpp()[i * (1 << n) + j]),
                    0, eps);
            }
        }
        delete mixture;
    }
    {
        DensityMatrixCpu state1(n);
        state1.set_Haar_random_state();
        QuantumState state2(n);
        state2.set_Haar_random_state();
        CPPCTYPE coef1(random.uniform(), random.uniform());
        CPPCTYPE coef2(random.uniform(), random.uniform());
        DensityMatrixCpu* mixture =
            state::make_mixture(coef1, &state1, coef2, &state2);
        for (UINT i = 0; i < (1 << n); ++i) {
            for (UINT j = 0; j < (1 << n); ++j) {
                CPPCTYPE val1 = state1.data_cpp()[i * (1 << n) + j];
                CPPCTYPE val2 =
                    state2.data_cpp()[i] * conj(state2.data_cpp()[j]);
                ASSERT_NEAR(abs(coef1 * val1 + coef2 * val2 -
                                mixture->data_cpp()[i * (1 << n) + j]),
                    0, eps);
            }
        }
        delete mixture;
    }
    {
        DensityMatrixCpu state1(n);
        state1.set_Haar_random_state();
        DensityMatrixCpu state2(n);
        state2.set_Haar_random_state();
        CPPCTYPE coef1(random.uniform(), random.uniform());
        CPPCTYPE coef2(random.uniform(), random.uniform());
        DensityMatrixCpu* mixture =
            state::make_mixture(coef1, &state1, coef2, &state2);
        for (UINT i = 0; i < (1 << n); ++i) {
            for (UINT j = 0; j < (1 << n); ++j) {
                CPPCTYPE val1 = state1.data_cpp()[i * (1 << n) + j];
                CPPCTYPE val2 = state2.data_cpp()[i * (1 << n) + j];
                ASSERT_NEAR(abs(coef1 * val1 + coef2 * val2 -
                                mixture->data_cpp()[i * (1 << n) + j]),
                    0, eps);
            }
        }
        delete mixture;
    }
}