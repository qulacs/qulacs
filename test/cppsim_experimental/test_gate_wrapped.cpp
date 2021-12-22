#include <gtest/gtest.h>

#include <cmath>
#include <cppsim_experimental/gate.hpp>
#include <cppsim_experimental/observable.hpp>
#include <cppsim_experimental/state.hpp>
#include <cppsim_experimental/utility.hpp>
#include <csim/update_ops.hpp>
#include <functional>

#include "../util/util.hpp"

TEST(GateTest, ProbabilisticGate) {
    auto gate1 = gate::X(0);
    auto gate2 = gate::X(1);
    auto gate3 = gate::X(2);
    auto prob_gate = QuantumGateWrapped::ProbabilisticGate(
        {gate1, gate2, gate3}, {0.25, 0.25, 0.25}, "", true);
    StateVectorCpu s(3);
    s.set_computational_basis(0);
    prob_gate->update_quantum_state(&s);
    delete prob_gate;
}

TEST(GateTest, CPTPGate) {
    ComplexMatrix mat0 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat1 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat2 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat3 = ComplexMatrix::Zero(4, 4);
    mat0(0, 0) = 1;
    mat1(1, 1) = 1;
    mat2(2, 2) = 1;
    mat3(3, 3) = 1;
    auto proj0 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat0);
    auto proj1 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat1);
    auto proj2 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat2);
    auto proj3 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat3);
    auto CPTP = QuantumGateWrapped::CPTP({proj0, proj1, proj2, proj3}, "", true);

    StateVectorCpu s(3);
    s.set_computational_basis(0);
    CPTP->update_quantum_state(&s);
    s.set_Haar_random_state();
    CPTP->update_quantum_state(&s);
    delete CPTP;
}

TEST(GateTest, InstrumentGate) {
    ComplexMatrix mat0 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat1 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat2 = ComplexMatrix::Zero(4, 4);
    ComplexMatrix mat3 = ComplexMatrix::Zero(4, 4);
    mat0(0, 0) = 1;
    mat1(1, 1) = 1;
    mat2(2, 2) = 1;
    mat3(3, 3) = 1;
    auto proj0 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat0);
    auto proj1 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat1);
    auto proj2 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat2);
    auto proj3 = QuantumGateBasic::DenseMatrixGate({0, 1}, mat3);
    std::string reg_name = "meas";
    auto Inst = QuantumGateWrapped::Instrument(
        {proj0, proj1, proj2, proj3}, reg_name, true);

    StateVectorCpu s(3);
    s.set_computational_basis(2);
    Inst->update_quantum_state(&s);
    UINT res1 = s.get_classical_value(reg_name);
    ASSERT_EQ(res1, 2);
    s.set_Haar_random_state();
    Inst->update_quantum_state(&s);
    UINT res2 = s.get_classical_value(reg_name);
    delete Inst;
}

TEST(GateTest, TestNoise) {
    const UINT n = 10;
    StateVectorCpu state(n);
    Random random;
    auto bitflip = gate::BitFlipNoise(0, random.uniform());
    auto dephase = gate::DephasingNoise(0, random.uniform());
    auto independetxz = gate::IndependentXZNoise(0, random.uniform());
    auto depolarizing = gate::DepolarizingNoise(0, random.uniform());
    auto amp_damp = gate::AmplitudeDampingNoise(0, random.uniform());
    auto measurement = gate::Measurement(0, "test");
    bitflip->update_quantum_state(&state);
    dephase->update_quantum_state(&state);
    independetxz->update_quantum_state(&state);
    depolarizing->update_quantum_state(&state);
    amp_damp->update_quantum_state(&state);
    measurement->update_quantum_state(&state);
    delete bitflip;
    delete dephase;
    delete independetxz;
    delete depolarizing;
    delete amp_damp;
    delete measurement;
}

/*
TEST(GateTest, AdaptiveGate) {
    auto x = gate::X(0);
    auto adaptive = gate::Adaptive(
        x, [](const std::vector<UINT>& vec) { return vec[2] == 1; });
    StateVectorCpu s(1);
    s.set_computational_basis(0);
    s.set_classical_value(2, 1);
    adaptive->update_quantum_state(&s);
    s.set_classical_value(2, 0);
    adaptive->update_quantum_state(&s);
    delete adaptive;
}
*/

/*
TEST(GateTest, GateAdd) {
    auto g1 = gate::X(0);
    auto g2 = gate::X(0);
    auto g3 = gate::X(1);
    auto g4 = gate::CNOT(0, 1);

    auto a1 = gate::add(g1, g2);
    auto a2 = gate::add(g1, g3);
    auto a3 = gate::add(g1, g4);
    auto a4 = gate::add(g3, g4);
    auto a5 = gate::add(gate::P0(0), gate::P1(0));
    auto a6 = gate::add(gate::merge(gate::P0(0), gate::P0(1)),
        gate::merge(gate::P1(0), gate::P1(1)));
    // TODO assert matrix element
}

TEST(GateTest, RandomUnitaryGate) {
    double eps = 1e-14;
    for (UINT qubit_count = 1; qubit_count < 5; ++qubit_count) {
        ITYPE dim = 1ULL << qubit_count;
        std::vector<UINT> target_qubit_list;
        for (UINT i = 0; i < qubit_count; ++i) {
            target_qubit_list.push_back(i);
        }
        auto gate = gate::RandomUnitary(target_qubit_list);
        ComplexMatrix cm;
        gate->set_matrix(cm);
        auto eye = cm * cm.adjoint();
        for (int i = 0; i < dim; ++i) {
            for (int j = 0; j < dim; ++j) {
                if (i == j) {
                    ASSERT_NEAR(abs(eye(i, j)), 1, eps);
                } else {
                    ASSERT_NEAR(abs(eye(i, j)), 0, eps);
                }
            }
        }
    }
}

TEST(GateTest, ReversibleBooleanGate) {
    const double eps = 1e-14;
    std::function<ITYPE(ITYPE, ITYPE)> func =
        [](ITYPE index, ITYPE dim) -> ITYPE { return (index + 1) % dim; };
    std::vector<UINT> target_qubit = {2, 0};
    auto gate = gate::ReversibleBoolean(target_qubit, func);
    ComplexMatrix cm;
    gate->set_matrix(cm);
    StateVectorCpu state(3);
    gate->update_quantum_state(&state);
    ASSERT_NEAR(abs(state.data_cpp()[4] - 1.), 0, eps);
    gate->update_quantum_state(&state);
    ASSERT_NEAR(abs(state.data_cpp()[1] - 1.), 0, eps);
    gate->update_quantum_state(&state);
    ASSERT_NEAR(abs(state.data_cpp()[5] - 1.), 0, eps);
    gate->update_quantum_state(&state);
    ASSERT_NEAR(abs(state.data_cpp()[0] - 1.), 0, eps);
}
*/

/*
TEST(GateTest, DuplicateIndex) {
    {
        auto gate1 = gate::CNOT(10, 13);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::CNOT(21, 21);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::CZ(10, 13);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::CZ(21, 21);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::SWAP(10, 13);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::SWAP(21, 21);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::Pauli({2, 1, 0, 3, 7, 9, 4}, {0, 0, 0, 0, 0, 0, 0});
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::Pauli({0, 1, 3, 1, 5, 6, 2}, {0, 0, 0, 0, 0, 0, 0});
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::PauliRotation(
            {2, 1, 0, 3, 7, 9, 4}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::PauliRotation(
            {0, 1, 3, 1, 5, 6, 2}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::DenseMatrix({10, 13}, ComplexMatrix::Identity(4, 4));
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::DenseMatrix({21, 21}, ComplexMatrix::Identity(4, 4));
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto matrix = SparseComplexMatrix(4, 4);
        matrix.setIdentity();
        auto gate1 = gate::SparseMatrix({10, 13}, matrix);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::SparseMatrix({21, 21}, matrix);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::RandomUnitary({10, 13});
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::RandomUnitary({21, 21});
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto ident = [](ITYPE a, ITYPE dim) { return a; };
        auto gate1 = gate::ReversibleBoolean({10, 13}, ident);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::ReversibleBoolean({21, 21}, ident);
        ASSERT_EQ(NULL, gate2);
    }
    {
        auto gate1 = gate::TwoQubitDepolarizingNoise(10, 13, 0.1);
        EXPECT_TRUE(gate1 != NULL);
        delete gate1;
        auto gate2 = gate::TwoQubitDepolarizingNoise(21, 21, 0.1);
        ASSERT_EQ(NULL, gate2);
    }
}
*/