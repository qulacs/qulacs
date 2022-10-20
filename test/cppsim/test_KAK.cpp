#include <gtest/gtest.h>

#include <Eigen/Eigen>
#include <cppsim/circuit.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_matrix.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/matrix_decomposition.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state_dm.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>
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

        delete random_gate;
    }
}

TEST(KAKTest, random1x1bit) {
    for (int i = 0; i < 5; i++) {
        QuantumGateBase* random_gate0 = gate::RandomUnitary({0});
        QuantumGateBase* random_gate1 = gate::RandomUnitary({1});
        QuantumGateBase* random_gate_merged =
            gate::merge(random_gate0, random_gate1);
        auto KAK_ret = KAK_decomposition(random_gate_merged, {0, 1});

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
        random_gate_merged->update_quantum_state(&stateA);
        circuit.update_quantum_state(&stateB);
        double inpro = abs(state::inner_product(&stateA, &stateB));
        ASSERT_NEAR(inpro, 1.0, 0.001);

        delete random_gate_merged;
        delete random_gate0;
        delete random_gate1;
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

        delete random_gate;
    }
}

TEST(CSDTest, random4bit) {
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

    delete random_gate;
}

TEST(CSDTest, empty3gate) {
    const UINT n_qubits = 3;

    std::vector<QuantumGateBase*> identity_gate_list;
    for (UINT i = 0; i < n_qubits; ++i) {
        identity_gate_list.push_back(gate::Identity(i));
    }

    QuantumGateBase* identity_gate_merged = gate::merge(identity_gate_list);
    auto CSD_ret = CSD(identity_gate_merged);

    QuantumCircuit circuit(4);
    for (auto it : CSD_ret) {
        circuit.add_gate(it);
    }
    QuantumState stateA(4);
    stateA.set_Haar_random_state();
    QuantumState stateB(4);
    stateB.load(&stateA);
    identity_gate_merged->update_quantum_state(&stateA);
    circuit.update_quantum_state(&stateB);
    double inpro = abs(state::inner_product(&stateA, &stateB));
    ASSERT_NEAR(inpro, 1.0, 0.001);

    delete identity_gate_merged;
    for (UINT i = 0; i < n_qubits; ++i) {
        delete identity_gate_list[i];
    }
}
