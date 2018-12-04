#include <gtest/gtest.h>
#include "../util/util.h"
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>

TEST(StateTest, GenerateAndRelease) {    
    UINT n = 10;
    double eps = 1e-14;
    QuantumState state(n);
    ASSERT_EQ(state.qubit_count, n);
    ASSERT_EQ(state.dim, 1ULL << n);
    state.set_zero_state();
    for (UINT i = 0; i < state.dim; ++i) {
        if (i == 0) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
        else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64()%state.dim;
        state.set_computational_basis(basis);
        for (UINT i = 0; i < state.dim; ++i) {
            if (i == basis) ASSERT_NEAR(abs(state.data_cpp()[i] - 1.), 0, eps);
            else ASSERT_NEAR(abs(state.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state.set_Haar_random_state();
        ASSERT_NEAR(state.get_norm(),1.,eps);
    }
}

TEST(StateTest, Sampling) {
    UINT n = 10;
    QuantumState state(n);
    state.set_Haar_random_state();
    state.set_computational_basis(100);
    auto res1 = state.sampling(1024);
    state.set_computational_basis(100);
    auto res2 = state.sampling(1024);
}