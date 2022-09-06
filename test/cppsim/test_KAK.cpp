#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <cppsim/KAK.hpp>
#include <cppsim/circuit.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>
#include <fstream>
#include <functional>

#include "../util/util.hpp"

using namespace Eigen;
TEST(KAKTest, random2bit) {
    for (int i = 0; i < 5; i++) {
        QuantumGateBase* random_gate = gate::RandomUnitary({0, 1});
        auto KAK_ret = KAK_decomposition(random_gate, {0, 1});

        QuantumCircuit circuit(2);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[1]);

        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {1, 1}, KAK_ret.interaction_coefficients[0]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {2, 2}, KAK_ret.interaction_coefficients[1]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {3, 3}, KAK_ret.interaction_coefficients[2]));
        circuit.add_gate(KAK_ret.single_qubit_operations_after[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_after[1]);
        QuantumState stateA(2);
        stateA.set_Haar_random_state();
        QuantumState stateB(2);
        stateB.load(&stateA);
        random_gate->update_quantum_state(&stateA);
        circuit.update_quantum_state(&stateB);
        double inpro = abs(state::inner_product(&stateA, &stateB));
        ASSERT_NEAR(inpro, 1.0, 0.001);
    }
}

TEST(KAKTest, random1x1bit) {
    for (int i = 0; i < 5; i++) {
        QuantumGateBase* random_gate =
            gate::merge(gate::RandomUnitary({0}), gate::RandomUnitary({1}));
        auto KAK_ret = KAK_decomposition(random_gate, {0, 1});

        QuantumCircuit circuit(2);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[1]);

        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {1, 1}, KAK_ret.interaction_coefficients[0]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {2, 2}, KAK_ret.interaction_coefficients[1]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {3, 3}, KAK_ret.interaction_coefficients[2]));
        circuit.add_gate(KAK_ret.single_qubit_operations_after[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_after[1]);
        QuantumState stateA(2);
        stateA.set_Haar_random_state();
        QuantumState stateB(2);
        stateB.load(&stateA);
        random_gate->update_quantum_state(&stateA);
        circuit.update_quantum_state(&stateB);
        double inpro = abs(state::inner_product(&stateA, &stateB));
        ASSERT_NEAR(inpro, 1.0, 0.001);
    }
}

TEST(KAKTest, CXgate) {
    for (int i = 0; i < 5; i++) {
        QuantumGateBase* random_gate = gate::CNOT(0, 1);

        auto KAK_ret = KAK_decomposition(random_gate, {0, 1});

        QuantumCircuit circuit(2);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_before[1]);

        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {1, 1}, KAK_ret.interaction_coefficients[0]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {2, 2}, KAK_ret.interaction_coefficients[1]));
        circuit.add_gate(gate::PauliRotation(
            {0, 1}, {3, 3}, KAK_ret.interaction_coefficients[2]));
        circuit.add_gate(KAK_ret.single_qubit_operations_after[0]);
        circuit.add_gate(KAK_ret.single_qubit_operations_after[1]);
        QuantumState stateA(2);
        stateA.set_Haar_random_state();
        QuantumState stateB(2);
        stateB.load(&stateA);
        random_gate->update_quantum_state(&stateA);
        circuit.update_quantum_state(&stateB);
        double inpro = abs(state::inner_product(&stateA, &stateB));
        ASSERT_NEAR(inpro, 1.0, 0.001);
    }
}

TEST(CSDTest, random4bit) {
    // QuantumGateBase* random_gate = gate::DenseMatrix({0, 1}, CXmat);
    QuantumGateBase* random_gate = gate::RandomUnitary({0, 1, 2, 3});

    auto CSD_ret = CSD(random_gate);

    QuantumCircuit circuit(4);
    for (auto it : CSD_ret) {
        // std::cerr << (*it) << std::endl;
        circuit.add_gate(it);
    }
    QuantumState stateA(4);
    stateA.set_Haar_random_state();
    QuantumState stateB(4);
    stateB.load(&stateA);
    random_gate->update_quantum_state(&stateA);
    circuit.update_quantum_state(&stateB);
    double inpro = abs(state::inner_product(&stateA, &stateB));
    ASSERT_NEAR(inpro, 1.0, 0.001);
}

TEST(CSDTest, empty3gate) {
    // QuantumGateBase* random_gate = gate::DenseMatrix({0, 1}, CXmat);
    QuantumGateBase* random_gate =
        gate::merge(gate::merge(gate::Identity({0}), gate::Identity({1})),
            gate::Identity({2}));
    auto CSD_ret = CSD(random_gate);

    QuantumCircuit circuit(4);
    for (auto it : CSD_ret) {
        // std::cerr << (*it) << std::endl;
        circuit.add_gate(it);
    }
    QuantumState stateA(4);
    stateA.set_Haar_random_state();
    QuantumState stateB(4);
    stateB.load(&stateA);
    random_gate->update_quantum_state(&stateA);
    circuit.update_quantum_state(&stateB);
    double inpro = abs(state::inner_product(&stateA, &stateB));
    ASSERT_NEAR(inpro, 1.0, 0.001);
}