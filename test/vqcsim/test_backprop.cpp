#include <gtest/gtest.h>

#include <cppsim/circuit.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/gate_merge.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <iostream>
#include <vqcsim/GradCalculator.hpp>
#include <vqcsim/parametric_circuit.hpp>
using namespace std;
TEST(Backprop, BackpropCircuit) {
    ParametricQuantumCircuit kairo(3);
    kairo.add_parametric_RX_gate(0, 2.2);
    kairo.add_parametric_RY_gate(1, 0);
    kairo.add_gate(gate::CNOT(0, 2));
    kairo.add_parametric_RZ_gate(2, 1.4);
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RY_gate(0, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RZ_gate(1, -1);
    kairo.add_gate(gate::H(0));
    kairo.add_gate(gate::CNOT(2, 0));
    kairo.add_parametric_RX_gate(2, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_parametric_RZ_gate(0, 1);
    kairo.add_gate(gate::CNOT(0, 1));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RX_gate(1, -1);
    kairo.add_parametric_RY_gate(2, 1);

    Observable observable(3);
    observable.add_operator(1, "X 0 Z 2");
    observable.add_operator(1.2, "Y 1");
    observable.add_operator(1.5, "Z 2");

    vector<double> kaku = {2.2, 0, 1.4, 1, -1, 1, 1, -1, 1};
    GradCalculator wrakln;
    auto bibun = wrakln.calculate_grad(kairo, observable, kaku);

    auto bk = kairo.backprop(&observable);
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(bk[i], bibun[i].real(), 1e-10);
    }
}

TEST(Backprop, BackpropCircuitInpro) {
    ParametricQuantumCircuit kairo(3);
    kairo.add_parametric_RX_gate(0, 2.2);
    kairo.add_parametric_RY_gate(1, 0);
    kairo.add_gate(gate::CNOT(0, 2));
    kairo.add_parametric_RZ_gate(2, 1.4);
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RY_gate(0, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RZ_gate(1, -1);
    kairo.add_gate(gate::H(0));
    kairo.add_gate(gate::CNOT(2, 0));
    kairo.add_parametric_RX_gate(2, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_parametric_RZ_gate(0, 1);
    kairo.add_gate(gate::CNOT(0, 1));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RX_gate(1, -1);
    kairo.add_parametric_RY_gate(2, 1);
    // 回路に適当にゲートを加える

    std::vector<CPPCTYPE> state_hai = {
        1.0, 0.5, 3.0, -0.2, -2.0, 1.0, 0.7, 3.0};
    QuantumState state_soku(3);
    state_soku.load(state_hai);

    QuantumState Astate(3);

    auto bk = kairo.backprop_inner_product(&state_soku);
    state_soku.load(state_hai);
    vector<double> kaku = {2.2, 0, 1.4, 1, -1, 1, 1, -1, 1};

    Astate.set_zero_state();
    kairo.update_quantum_state(&Astate);
    CPPCTYPE mto_sco = state::inner_product(&state_soku, &Astate);
    for (int h = 0; h < 9; h++) {
        kairo.set_parameter(h, kaku[h] + 0.0001);
        Astate.set_zero_state();
        kairo.update_quantum_state(&Astate);
        CPPCTYPE gen_sco = state::inner_product(&state_soku, &Astate);

        ASSERT_NEAR(((gen_sco - mto_sco) * 10000.0).real(), bk[h], 1e-2);

        kairo.set_parameter(h, kaku[h]);
    }
}

TEST(Backprop_multicpu, BackpropCircuitInpro) {
    ParametricQuantumCircuit kairo(3);
    kairo.add_parametric_RX_gate(0, 2.2);
    kairo.add_parametric_RY_gate(1, 0);
    kairo.add_gate(gate::CNOT(0, 2));
    kairo.add_parametric_RZ_gate(2, 1.4);
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RY_gate(0, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RZ_gate(1, -1);
    kairo.add_gate(gate::H(0));
    kairo.add_gate(gate::CNOT(2, 0));
    kairo.add_parametric_RX_gate(2, 1);
    kairo.add_gate(gate::CNOT(1, 0));
    kairo.add_parametric_RZ_gate(0, 1);
    kairo.add_gate(gate::CNOT(0, 1));
    kairo.add_gate(gate::H(1));
    kairo.add_parametric_RX_gate(1, -1);
    kairo.add_parametric_RY_gate(2, 1);
    // 回路に適当にゲートを加える

    std::vector<CPPCTYPE> state_hai = {
        1.0, 0.5, 3.0, -0.2, -2.0, 1.0, 0.7, 3.0};

    QuantumState state_soku_cpu(3, 0);
    QuantumState state_soku(3, 1);
    // Loading from full-size-vector into a distributed-SV is not supported.
    // Loading from undistributed-SV into a distributed-SV is supported.
    state_soku_cpu.load(state_hai);
    state_soku.load(&state_soku_cpu);

    QuantumState Astate(3, true);

    auto bk = kairo.backprop_inner_product(&state_soku);
    state_soku_cpu.load(state_hai);
    state_soku.load(&state_soku_cpu);
    vector<double> kaku = {2.2, 0, 1.4, 1, -1, 1, 1, -1, 1};

    Astate.set_zero_state();
    kairo.update_quantum_state(&Astate);
    CPPCTYPE mto_sco = state::inner_product(&state_soku, &Astate);
    for (int h = 0; h < 9; h++) {
        kairo.set_parameter(h, kaku[h] + 0.0001);
        Astate.set_zero_state();
        kairo.update_quantum_state(&Astate);
        CPPCTYPE gen_sco = state::inner_product(&state_soku, &Astate);

        ASSERT_NEAR(((gen_sco - mto_sco) * 10000.0).real(), bk[h], 1e-2);

        kairo.set_parameter(h, kaku[h]);
    }
}

// PauliRotationのBackpropTest
TEST(Backprop, BackpropPauliRotationCircuit) {
    ParametricQuantumCircuit kairo(2);
    vector<double> kaku = {2.2, 0, 1.4, 0.8, -0.4, 1.2, 0.7, -1, 1.3, 0.5, -0.8,
        -2.1, 1.9, 2.1, 0.3};

    int ind = 0;
    kairo.add_parametric_RX_gate(0, kaku[ind++]);
    kairo.add_parametric_RY_gate(0, kaku[ind++]);
    kairo.add_parametric_RZ_gate(0, kaku[ind++]);
    kairo.add_parametric_RX_gate(1, kaku[ind++]);
    kairo.add_parametric_RY_gate(1, kaku[ind++]);
    kairo.add_parametric_RZ_gate(1, kaku[ind++]);

    // PauliRotation
    // Rxx
    kairo.add_parametric_multi_Pauli_rotation_gate({0, 1}, {1, 1}, kaku[ind++]);
    // Ryy
    kairo.add_parametric_multi_Pauli_rotation_gate({0, 1}, {2, 2}, kaku[ind++]);
    // Rzz
    kairo.add_parametric_multi_Pauli_rotation_gate({0, 1}, {3, 3}, kaku[ind++]);

    kairo.add_parametric_RX_gate(0, kaku[ind++]);
    kairo.add_parametric_RY_gate(0, kaku[ind++]);
    kairo.add_parametric_RZ_gate(0, kaku[ind++]);
    kairo.add_parametric_RX_gate(1, kaku[ind++]);
    kairo.add_parametric_RY_gate(1, kaku[ind++]);
    kairo.add_parametric_RZ_gate(1, kaku[ind++]);

    Observable observable(2);
    observable.add_operator(1, "X 0");

    GradCalculator wrakln;
    auto bibun = wrakln.calculate_grad(kairo, observable, kaku);

    auto bk = kairo.backprop(&observable);
    // 数値微分との比較
    for (int i = 0; i < 9; i++) {
        ASSERT_NEAR(bk[i], bibun[i].real(), 1e-10);
    }
}
