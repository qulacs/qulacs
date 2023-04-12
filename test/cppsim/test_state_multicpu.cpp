#ifdef _USE_MPI
#include <gtest/gtest.h>

#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <csim/MPIutil.hpp>

#include "../util/util.hpp"

TEST(StateTest_multicpu, GenerateAndRelease) {
    UINT n = 10;
    UINT mpirank, mpisize, global_qubit, local_qubit;
    ITYPE part_dim;

    QuantumState state_multicpu(n, true);
    if (state_multicpu.get_device_name() == "multi-cpu") {
        MPIutil &mpiutil = MPIutil::get_inst();
        mpirank = mpiutil.get_rank();
        mpisize = mpiutil.get_size();
        global_qubit = std::log2(mpisize);
        local_qubit = n - global_qubit;
        part_dim = (1ULL << n) / mpisize;
    } else {
        mpirank = 0;
        mpisize = 1;
        global_qubit = 0;
        local_qubit = n;
        part_dim = 1ULL << n;
    }

    ASSERT_EQ(state_multicpu.qubit_count, n);
    ASSERT_EQ(state_multicpu.dim, part_dim);
    state_multicpu.set_zero_state();
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        if (i == 0 && mpirank == 0)
            ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] - 1.), 0, eps);
        else
            ASSERT_NEAR(abs(state_multicpu.data_cpp()[i]), 0, eps);
    }
    Random random;
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        ITYPE basis = random.int64() % state_multicpu.dim;
        state_multicpu.set_computational_basis(basis);
        for (UINT i = 0; i < state_multicpu.dim; ++i) {
            if (i == (basis % (1ULL << local_qubit)) &&
                basis >> local_qubit == mpirank)
                ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] - 1.), 0, eps);
            else
                ASSERT_NEAR(abs(state_multicpu.data_cpp()[i]), 0, eps);
        }
    }
    for (UINT repeat = 0; repeat < 10; ++repeat) {
        state_multicpu.set_Haar_random_state();
        ASSERT_NEAR(state_multicpu.get_squared_norm(), 1., eps);
    }
}

TEST(StateTest_multicpu, setHaarRandomState) {
    UINT n = 10;
    UINT mpirank, mpisize, global_qubit, local_qubit;
    ITYPE part_dim, offs;

    QuantumState state_multicpu(n, true);
    QuantumState state_singlecpu(n, false);

    if (state_multicpu.get_device_name() == "multi-cpu") {
        MPIutil &mpiutil = MPIutil::get_inst();
        mpirank = mpiutil.get_rank();
        mpisize = mpiutil.get_size();
        global_qubit = std::log2(mpisize);
        local_qubit = n - global_qubit;
        part_dim = (1ULL << n) / mpisize;
        offs = part_dim * mpirank;
    } else {
        mpirank = 0;
        mpisize = 1;
        global_qubit = 0;
        local_qubit = n;
        part_dim = 1ULL << n;
        offs = 0ULL;
    }

    state_multicpu.set_computational_basis(600);
    state_singlecpu.load(&state_multicpu);
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] -
                        state_singlecpu.data_cpp()[i + offs]),
            0, eps);
    }

    state_singlecpu.set_computational_basis(600);
    state_multicpu.load(&state_singlecpu);
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] -
                        state_singlecpu.data_cpp()[i + offs]),
            0, eps);
    }

    state_multicpu.set_Haar_random_state();
    state_singlecpu.load(&state_multicpu);
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] -
                        state_singlecpu.data_cpp()[i + offs]),
            0, eps);
    }

    state_singlecpu.set_Haar_random_state();
    state_multicpu.load(&state_singlecpu);
    for (UINT i = 0; i < state_multicpu.dim; ++i) {
        ASSERT_NEAR(abs(state_multicpu.data_cpp()[i] -
                        state_singlecpu.data_cpp()[i + offs]),
            0, eps);
    }
}

TEST(StateTest_multicpu, SamplingComputationalBasis) {
    const UINT n = 10;
    const UINT nshot = 1024;
    QuantumState state(n, true);
    state.set_computational_basis(100);
    auto res = state.sampling(nshot);
    for (UINT i = 0; i < nshot; ++i) {
        ASSERT_TRUE(res[i] == 100);
    }
}

TEST(StateTest_multicpu, SamplingSuperpositionState) {
    const UINT n = 10;
    const UINT nshot = 1024;
    QuantumState state(n, true);
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
    for (UINT i = 0; i < 4; i++) {
        ASSERT_GT(cnt[i + 1], cnt[i]);
    }
}

TEST(StateTest_multicpu, InnerProductSimple) {
    const UINT n = 10;
    QuantumState state_bra_s(n);
    QuantumState state_ket_s(n);
    QuantumState state_bra_d(n, true);
    QuantumState state_ket_d(n, true);
    state_bra_s.set_Haar_random_state(2000);
    state_ket_d.set_Haar_random_state(2001);
    state_bra_d.load(&state_bra_s);
    state_ket_s.load(&state_ket_d);

    double result_s_s =
        std::abs(state::inner_product(&state_bra_s, &state_ket_s));
    double result_s_d =
        std::abs(state::inner_product(&state_bra_s, &state_ket_d));
    double result_d_s =
        std::abs(state::inner_product(&state_bra_d, &state_ket_s));
    double result_d_d =
        std::abs(state::inner_product(&state_bra_d, &state_ket_d));

    ASSERT_NEAR(result_s_s, result_s_d, eps);
    ASSERT_NEAR(result_s_s, result_d_s, eps);
    ASSERT_NEAR(result_s_s, result_d_d, eps);
}
#endif