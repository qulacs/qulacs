#include <gtest/gtest.h>

#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>

#include "../util/util.hpp"

TEST(StateTest, GenerateAndRelease) {
    UINT n = 10;

    QuantumState state(n);
    ASSERT_EQ(state.qubit_count, n);
    ASSERT_EQ(state.dim, 1ULL << n);
    state.set_zero_state();
    for (UINT i = 0; i < state.dim; ++i) {
        if (i == 0)
            ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
        else
            ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64() % state.dim;
        state.set_computational_basis(basis);
        for (UINT i = 0; i < state.dim; ++i) {
            if (i == basis)
                ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
            else
                ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state.set_Haar_random_state();
        ASSERT_NEAR(state.get_squared_norm(), 1., eps);
    }
}

TEST(StateTest, SamplingComputationalBasis) {
    const UINT n = 10;
    const UINT nshot = 1024;
    QuantumState state(n);
    state.set_computational_basis(100);
    auto res = state.sampling(nshot);
    for (UINT i = 0; i < nshot; ++i) {
        ASSERT_TRUE(res[i] == 100);
    }
}

TEST(StateTest, SamplingSuperpositionState) {
    const UINT n = 10;
    const UINT nshot = 1024;
    const UINT test_count = 10;
    UINT pass_count = 0;
    for (UINT test_i = 0; test_i < test_count; test_i++) {
        QuantumState state(n);
        state.set_computational_basis(0);
        for (ITYPE i = 1; i <= 4; ++i) {
            QuantumState tmp_state(n);
            tmp_state.set_computational_basis(i);
            state.add_state_with_coef_single_thread(1 << i, &tmp_state);
        }
        state.normalize_single_thread(state.get_squared_norm_single_thread());
        auto res = state.sampling(nshot);
        std::array<UINT, 5> cnt = {};
        for (UINT i = 0; i < nshot; ++i) {
            ASSERT_GE(res[i], 0);
            ASSERT_LE(res[i], 4);
            cnt[res[i]] += 1;
        }
        bool pass = true;
        for (UINT i = 0; i < 4; i++) {
            std::string err_message = _CHECK_GT(cnt[i + 1], cnt[i]);
            if (err_message != "") {
                pass = false;
                std::cerr << err_message;
            }
        }
        if (pass) pass_count++;
    }
    ASSERT_GE(pass_count, test_count - 1);
}

TEST(StateTest, SetState) {
    const UINT n = 10;
    QuantumState state(n);
    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        const auto d = static_cast<double>(i);
        state_vector[i] = d + std::complex<double>(0, 1) * (d + 0.1);
    }
    state.load(state_vector);
    for (ITYPE i = 0; i < dim; ++i) {
        ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
        ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
    }
}

TEST(StateTest, HaarRandomStateNorm) {
    const UINT n_max = 5, m = 10;
    for (UINT n = 1; n <= n_max; n++) {
        for (UINT i = 0; i < m; i++) {
            QuantumState state(n);
            state.set_Haar_random_state();
            ASSERT_NEAR(state.get_squared_norm_single_thread(), 1., 1e-10);
        }
    }
}

bool same_state(QuantumState* s1, QuantumState* s2) {
    assert(s1->qubit_count == s2->qubit_count);
    auto s1_data = s1->data_cpp();
    auto s2_data = s2->data_cpp();
    for (ITYPE i = 0; i < s1->dim; ++i) {
        if (abs(s1_data[i] - s2_data[i]) > eps) return false;
    }
    return true;
};

TEST(StateTest, HaarRandomStateWithoutSeed) {
    const UINT n = 10, m = 5;
    std::vector<QuantumState*> states(m);
    for (UINT i = 0; i < m; ++i) {
        states[i] = new QuantumState(n);
        states[i]->set_Haar_random_state();
    }
    for (UINT i = 0; i < m - 1; ++i) {
        for (UINT j = i + 1; j < m; ++j) {
            ASSERT_FALSE(same_state(states[i], states[j]));
        }
    }
    for (UINT i = 0; i < m; i++) {
        delete states[i];
    }
}

TEST(StateTest, StateTest_HaarRandomStateSameSeed) {
    const UINT n = 10, m = 5;
    for (UINT i = 0; i < m; ++i) {
        QuantumState state1(n), state2(n);
        state1.set_Haar_random_state(i);
        state2.set_Haar_random_state(i);
        ASSERT_TRUE(same_state(&state1, &state2));
    }
}

TEST(StateTest, GetZeroProbability) {
    const UINT n = 10;

    QuantumState state(n);
    state.set_computational_basis(1);
    for (ITYPE i = 2; i <= 10; ++i) {
        QuantumState tmp_state(n);
        tmp_state.set_computational_basis(i);
        state.add_state_with_coef_single_thread(std::sqrt(i), &tmp_state);
    }
    state.normalize_single_thread(state.get_squared_norm_single_thread());
    ASSERT_NEAR(state.get_zero_probability(0), 30.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(1), 27.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(2), 33.0 / 55.0, eps);
    ASSERT_NEAR(state.get_zero_probability(3), 28.0 / 55.0, eps);
}

TEST(StateTest, GetMarginalProbability) {
    const UINT n = 2;
    const ITYPE dim = 1 << n;
    QuantumState state(n);
    state.set_Haar_random_state();
    std::vector<double> probs;
    for (ITYPE i = 0; i < dim; ++i) {
        probs.push_back(pow(abs(state.data_cpp()[i]), 2));
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

TEST(StateTest, AddState) {
    const UINT n = 10;
    QuantumState state1(n);
    QuantumState state2(n);
    state1.set_Haar_random_state();
    state2.set_Haar_random_state();

    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector1(dim);
    std::vector<std::complex<double>> state_vector2(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        state_vector1[i] = state1.data_cpp()[i];
        state_vector2[i] = state2.data_cpp()[i];
    }

    state1.add_state(&state2);

    for (ITYPE i = 0; i < dim; ++i) {
        ASSERT_NEAR(state1.data_cpp()[i].real(),
            state_vector1[i].real() + state_vector2[i].real(), eps);
        ASSERT_NEAR(state1.data_cpp()[i].imag(),
            state_vector1[i].imag() + state_vector2[i].imag(), eps);
        ASSERT_NEAR(state2.data_cpp()[i].real(), state_vector2[i].real(), eps);
        ASSERT_NEAR(state2.data_cpp()[i].imag(), state_vector2[i].imag(), eps);
    }
}

TEST(StateTest, AddStateWithCoef) {
    const std::complex<double> coef(2.5, 1.3);
    const UINT n = 10;
    QuantumState state1(n);
    QuantumState state2(n);
    state1.set_Haar_random_state();
    state2.set_Haar_random_state();

    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector1(dim);
    std::vector<std::complex<double>> state_vector2(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        state_vector1[i] = state1.data_cpp()[i];
        state_vector2[i] = state2.data_cpp()[i];
    }

    state1.add_state_with_coef(coef, &state2);

    for (ITYPE i = 0; i < dim; ++i) {
        ASSERT_NEAR(state1.data_cpp()[i].real(),
            state_vector1[i].real() + coef.real() * state_vector2[i].real() -
                coef.imag() * state_vector2[i].imag(),
            eps);
        ASSERT_NEAR(state1.data_cpp()[i].imag(),
            state_vector1[i].imag() + coef.real() * state_vector2[i].imag() +
                coef.imag() * state_vector2[i].real(),
            eps);
        ASSERT_NEAR(state2.data_cpp()[i].real(), state_vector2[i].real(), eps);
        ASSERT_NEAR(state2.data_cpp()[i].imag(), state_vector2[i].imag(), eps);
    }
}

TEST(StateTest, MultiplyCoef) {
    const UINT n = 10;
    const std::complex<double> coef(0.5, 0.2);

    QuantumState state(n);
    state.set_Haar_random_state();

    const ITYPE dim = 1ULL << n;
    std::vector<std::complex<double>> state_vector(dim);
    for (ITYPE i = 0; i < dim; ++i) {
        state_vector[i] = state.data_cpp()[i] * coef;
    }
    state.multiply_coef(coef);

    for (ITYPE i = 0; i < dim; ++i) {
        ASSERT_NEAR(state.data_cpp()[i].real(), state_vector[i].real(), eps);
        ASSERT_NEAR(state.data_cpp()[i].imag(), state_vector[i].imag(), eps);
    }
}

TEST(StateTest, TensorProduct) {
    const UINT n = 5;

    QuantumState state1(n), state2(n);
    state1.set_Haar_random_state();
    state2.set_Haar_random_state();

    QuantumState* state3 = state::tensor_product(&state1, &state2);
    for (ITYPE i = 0; i < state1.dim; ++i) {
        for (ITYPE j = 0; j < state2.dim; ++j) {
            ASSERT_NEAR(state3->data_cpp()[i * state2.dim + j].real(),
                (state1.data_cpp()[i] * state2.data_cpp()[j]).real(), eps);
            ASSERT_NEAR(state3->data_cpp()[i * state2.dim + j].imag(),
                (state1.data_cpp()[i] * state2.data_cpp()[j]).imag(), eps);
        }
    }
    delete state3;
}

TEST(StateTest, DropQubit) {
    const UINT n = 4;

    QuantumState state(n);
    state.set_Haar_random_state();
    QuantumState* state2 = state::drop_qubit(&state, {2, 0}, {0, 1});

    ASSERT_EQ(state2->dim, 4);
    int corr[] = {1, 3, 9, 11};
    for (ITYPE i = 0; i < state2->dim; ++i) {
        ASSERT_NEAR(state2->data_cpp()[i].real(),
            state.data_cpp()[corr[i]].real(), eps);
        ASSERT_NEAR(state2->data_cpp()[i].imag(),
            state.data_cpp()[corr[i]].imag(), eps);
    }
    delete state2;
}

TEST(StateTest, PermutateQubit) {
    const UINT n = 3;

    QuantumState state(n);
    state.set_Haar_random_state();
    QuantumState* state2 = state::permutate_qubit(&state, {1, 0, 2});

    int corr[] = {0, 2, 1, 3, 4, 6, 5, 7};
    for (ITYPE i = 0; i < state2->dim; ++i) {
        ASSERT_NEAR(state2->data_cpp()[i].real(),
            state.data_cpp()[corr[i]].real(), eps);
        ASSERT_NEAR(state2->data_cpp()[i].imag(),
            state.data_cpp()[corr[i]].imag(), eps);
    }
    delete state2;
}

TEST(StateTest, ZeroNormState) {
    const UINT n = 5;

    QuantumState state(n);
    state.set_Haar_random_state();
    state.set_zero_norm_state();
    std::complex<double>* result = state.data_cpp();
    for (int i = 0; i < (1 << n); ++i) {
        ASSERT_EQ(result[i], std::complex<double>(0, 0));
    }
}

TEST(StateTest, MakeSuperposition) {
    const UINT n = 4;
    QuantumState state1(n);
    state1.set_Haar_random_state();
    QuantumState state2(n);
    state2.set_Haar_random_state();
    Random random;
    CPPCTYPE coef1(random.uniform(), random.uniform());
    CPPCTYPE coef2(random.uniform(), random.uniform());
    QuantumState* superposition =
        state::make_superposition(coef1, &state1, coef2, &state2);
    for (UINT i = 0; i < (1 << n); ++i) {
        ASSERT_NEAR(
            abs(coef1 * state1.data_cpp()[i] + coef2 * state2.data_cpp()[i] -
                superposition->data_cpp()[i]),
            0, eps);
    }
    delete superposition;
}
