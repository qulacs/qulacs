#include <gtest/gtest.h>

#include <cmath>
#include <cppsim_experimental/gate.hpp>
#include <cppsim_experimental/observable.hpp>
#include <cppsim_experimental/state.hpp>
#include <cppsim_experimental/state_dm.hpp>
#include <cppsim_experimental/utility.hpp>
#include <csim/update_ops.hpp>
#include <functional>
#include <numeric>

#include "../util/util.hpp"

TEST(DensityMatrixGeneralGateTest, CheckProbabilisticGate) {
    const UINT n = 5;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;
    const UINT gate_count = 5;

    Random random;
    DensityMatrix state(n);

    std::vector<UINT> arr(n);
    std::iota(arr.begin(), arr.end(), 0);

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        // create dist
        std::vector<double> probs;
        for (UINT i = 0; i < gate_count; ++i) probs.push_back(random.uniform());
        double sum = std::accumulate(probs.begin(), probs.end(), 0.);
        for (UINT i = 0; i < gate_count; ++i) probs[i] /= sum;

        // create gate list
        std::vector<QuantumGateBase*> gate_list;
        for (UINT i = 0; i < gate_count; ++i) {
            auto gate = gate::RandomUnitary(arr);
            gate_list.push_back(gate);
        }
        auto prob_gate =
            QuantumGateWrapped::ProbabilisticGate(gate_list, probs);

        // update density matrix
        DensityMatrix dm(n);
        dm.set_Haar_random_state();

        // update by matrix reps
        ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
        for (UINT i = 0; i < gate_count; ++i) {
            ComplexMatrix gate_mat;
            gate_list[i]->get_matrix(gate_mat);
            ComplexMatrix dense_mat(dim, dim);
            for (ITYPE i = 0; i < dim; ++i)
                for (ITYPE j = 0; j < dim; ++j)
                    dense_mat(i, j) = dm.data_cpp()[i * dim + j];
            mat += probs[i] * gate_mat * dense_mat * gate_mat.adjoint();
        }
        prob_gate->update_quantum_state(&dm);

        // check equivalence
        for (ITYPE i = 0; i < dim; ++i)
            for (ITYPE j = 0; j < dim; ++j)
                ASSERT_NEAR(
                    abs(dm.data_cpp()[i * dim + j] - mat(i, j)), 0., eps);
        // check TP
        ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

        // release
        delete prob_gate;
        for (UINT i = 0; i < gate_count; ++i) {
            delete gate_list[i];
        }
    }
}

TEST(DensityMatrixGeneralGateTest, CheckCPTPMap) {
    const UINT n = 2;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;
    const UINT gate_count = 5;

    Random random;
    DensityMatrix state(n);

    std::vector<UINT> arr(n);
    std::iota(arr.begin(), arr.end(), 0);

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        // create dist
        std::vector<double> probs;
        for (UINT i = 0; i < gate_count; ++i) probs.push_back(random.uniform());
        double sum = std::accumulate(probs.begin(), probs.end(), 0.);
        for (UINT i = 0; i < gate_count; ++i) probs[i] /= sum;

        // create not TP gate list
        std::vector<QuantumGateBase*> gate_list;
        for (UINT i = 0; i < gate_count; ++i) {
            auto gate = gate::RandomUnitary(arr);
            gate->multiply_scalar(sqrt(probs[i]));
            gate_list.push_back(gate);
        }
        auto cptp_gate = QuantumGateWrapped::CPTP(gate_list);

        // update density matrix
        DensityMatrix dm(n);
        dm.set_Haar_random_state();

        // update by matrix reps
        ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
        for (UINT i = 0; i < gate_count; ++i) {
            ComplexMatrix gate_mat;
            gate_list[i]->get_matrix(gate_mat);
            ComplexMatrix dense_mat(dim, dim);
            for (ITYPE i = 0; i < dim; ++i)
                for (ITYPE j = 0; j < dim; ++j)
                    dense_mat(i, j) = dm.data_cpp()[i * dim + j];
            mat += gate_mat * dense_mat * gate_mat.adjoint();
        }
        cptp_gate->update_quantum_state(&dm);

        // check equivalence
        for (ITYPE i = 0; i < dim; ++i)
            for (ITYPE j = 0; j < dim; ++j)
                ASSERT_NEAR(
                    abs(dm.data_cpp()[i * dim + j] - mat(i, j)), 0., eps);
        // check TP
        ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

        // release
        delete cptp_gate;
        for (UINT i = 0; i < gate_count; ++i) {
            delete gate_list[i];
        }
    }
}

TEST(DensityMatrixGeneralGateTest, AmplitudeDampingTest) {
    const UINT n = 1;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;

    Random random;
    DensityMatrix state(n);

    std::vector<UINT> arr(n);
    std::iota(arr.begin(), arr.end(), 0);

    for (UINT repeat = 0; repeat < 10; ++repeat) {
        double prob = random.uniform();

        ComplexMatrix K0(2, 2), K1(2, 2);
        K0 << 1, 0, 0, sqrt(1 - prob);
        K1 << 0, sqrt(prob), 0, 0;

        auto gate0 = QuantumGateBasic::DenseMatrixGate(arr, K0);
        auto gate1 = QuantumGateBasic::DenseMatrixGate(arr, K1);
        std::vector<QuantumGateBase*> gate_list = {gate0, gate1};
        auto cptp_gate = QuantumGateWrapped::CPTP(gate_list);

        // update density matrix
        DensityMatrix dm(n);
        dm.set_Haar_random_state();

        // update by matrix reps
        ComplexMatrix mat = ComplexMatrix::Zero(dim, dim);
        for (UINT i = 0; i < gate_list.size(); ++i) {
            ComplexMatrix gate_mat;
            gate_list[i]->get_matrix(gate_mat);
            ComplexMatrix dense_mat(dim, dim);
            for (ITYPE i = 0; i < dim; ++i)
                for (ITYPE j = 0; j < dim; ++j)
                    dense_mat(i, j) = dm.data_cpp()[i * dim + j];
            mat += gate_mat * dense_mat * gate_mat.adjoint();
        }
        cptp_gate->update_quantum_state(&dm);

        // check equivalence
        for (ITYPE i = 0; i < dim; ++i)
            for (ITYPE j = 0; j < dim; ++j)
                ASSERT_NEAR(
                    abs(dm.data_cpp()[i * dim + j] - mat(i, j)), 0., eps);
        // check TP
        ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

        // release
        delete cptp_gate;
        for (UINT i = 0; i < gate_list.size(); ++i) {
            delete gate_list[i];
        }
    }
}

TEST(DensityMatrixGeneralGateTest, DepolarizingTest) {
    const UINT n = 1;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;
    double prob = 0.2;

    Random random;
    DensityMatrix state(n);

    // update density matrix
    DensityMatrix dm(n);
    dm.set_Haar_random_state();
    ComplexMatrix dense_mat(dim, dim);
    for (ITYPE i = 0; i < dim; ++i)
        for (ITYPE j = 0; j < dim; ++j)
            dense_mat(i, j) = dm.data_cpp()[i * dim + j];
    ASSERT_NEAR(dense_mat.norm(), 1., eps);
    ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

    auto conv_mat =
        dense_mat * (1 - prob) + prob / dim * ComplexMatrix::Identity(dim, dim);
    auto two_qubit_depolarizing = gate::DepolarizingNoise(0, prob * 3 / 4);
    two_qubit_depolarizing->update_quantum_state(&dm);
    ASSERT_NEAR(dense_mat.norm(), 1., eps);
    ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

    // std::cout << dense_mat << std::endl;
    // std::cout << dm << std::endl;

    // check equivalence
    for (ITYPE i = 0; i < dim; ++i)
        for (ITYPE j = 0; j < dim; ++j)
            ASSERT_NEAR(
                abs(dm.data_cpp()[i * dim + j] - conv_mat(i, j)), 0., eps);
    delete two_qubit_depolarizing;
}

TEST(DensityMatrixGeneralGateTest, TwoQubitDepolarizingTest) {
    const UINT n = 2;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;
    double prob = 0.2;

    Random random;
    DensityMatrix state(n);

    // update density matrix
    DensityMatrix dm(n);
    dm.set_Haar_random_state();
    ComplexMatrix dense_mat(dim, dim);
    for (ITYPE i = 0; i < dim; ++i)
        for (ITYPE j = 0; j < dim; ++j)
            dense_mat(i, j) = dm.data_cpp()[i * dim + j];
    ASSERT_NEAR(dense_mat.norm(), 1., eps);
    ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

    // std::cout << dense_mat << std::endl;
    // std::cout << dm << std::endl;
    auto conv_mat =
        dense_mat * (1 - prob) + prob / dim * ComplexMatrix::Identity(dim, dim);
    auto two_qubit_depolarizing =
        gate::TwoQubitDepolarizingNoise(0, 1, prob * 15 / 16);
    two_qubit_depolarizing->update_quantum_state(&dm);
    // std::cout << conv_mat << std::endl;
    // std::cout << dm << std::endl;
    ASSERT_NEAR(dense_mat.norm(), 1., eps);
    ASSERT_NEAR(dm.get_squared_norm(), 1., eps);

    // check equivalence
    for (ITYPE i = 0; i < dim; ++i)
        for (ITYPE j = 0; j < dim; ++j)
            ASSERT_NEAR(
                abs(dm.data_cpp()[i * dim + j] - conv_mat(i, j)), 0., eps);
    // check TP
    delete two_qubit_depolarizing;
}

TEST(DensityMatrixGeneralGateTest, NoiseSampling) {
    const UINT n = 1;
    const ITYPE dim = 1ULL << n;
    double eps = 1e-15;
    double prob = 0.2;

    Random random;
    DensityMatrix state(n);

    // update density matrix
    DensityMatrix dm(n);
    dm.set_Haar_random_state();
    std::vector<ITYPE> samples = dm.sampling(1000);
    for (UINT i = 0; i < samples.size(); ++i) {
        ASSERT_TRUE(samples[i] == 0 || samples[i] == 1);
    }
}
