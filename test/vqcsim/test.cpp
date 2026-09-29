#include <gtest/gtest.h>

#include <cppsim/exception.hpp>
#include <cppsim/gate_factory.hpp>
#include <cppsim/state_dm.hpp>
#include <vqcsim/GradCalculator.hpp>
#include <vqcsim/causalcone_simulator.hpp>
#include <vqcsim/parametric_circuit_builder.hpp>
#include <vqcsim/parametric_gate_factory.hpp>
#include <vqcsim/problem.hpp>
#include <vqcsim/solver.hpp>

#include "../util/util.hpp"

class ClsParametricNullUpdateGate
    : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricNullUpdateGate(UINT target_qubit_index, double angle)
        : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_name = "ParametricNullUpdate";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index));
    }
    virtual void set_matrix(ComplexMatrix& matrix) const override {}
    virtual QuantumGate_SingleParameter* copy() const override {
        return new ClsParametricNullUpdateGate(*this);
    };
};

TEST(ParametricGate, NullUpdateFunc) {
    ClsParametricNullUpdateGate gate(0, 0.);
    QuantumState state(1);
    ASSERT_THROW(
        gate.update_quantum_state(&state), UndefinedUpdateFuncException);
}

TEST(ParametricGate_multicpu, NullUpdateFunc) {
    ClsParametricNullUpdateGate gate(0, 0.);
    QuantumState state(3, true);
    ASSERT_THROW(
        gate.update_quantum_state(&state), UndefinedUpdateFuncException);
}

TEST(ParametricCircuit, GateApply) {
    const UINT n = 3;
    const UINT depth = 10;
    ParametricQuantumCircuit* circuit = new ParametricQuantumCircuit(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            circuit->add_parametric_RX_gate(i, random.uniform());
            circuit->add_parametric_RY_gate(i, random.uniform());
            circuit->add_parametric_RZ_gate(i, random.uniform());
        }
        for (UINT i = d % 2; i + 1 < n; i += 2) {
            circuit->add_parametric_multi_Pauli_rotation_gate(
                {i, i + 1}, {3, 3}, random.uniform());
        }
    }

    UINT param_count = circuit->get_parameter_count();
    for (UINT p = 0; p < param_count; ++p) {
        double current_angle = circuit->get_parameter(p);
        circuit->set_parameter(p, current_angle + random.uniform());
    }

    QuantumState state(n);
    circuit->update_quantum_state(&state);
    // std::cout << state << std::endl;
    // std::cout << circuit << std::endl;
    delete circuit;
}

TEST(ParametricCircuit_multicpu, GateApply) {
    const UINT n = 3;
    const UINT depth = 10;
    ParametricQuantumCircuit* circuit = new ParametricQuantumCircuit(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            circuit->add_parametric_RX_gate(i, random.uniform());
            circuit->add_parametric_RY_gate(i, random.uniform());
            circuit->add_parametric_RZ_gate(i, random.uniform());
        }
        for (UINT i = d % 2; i + 1 < n; i += 2) {
            circuit->add_parametric_multi_Pauli_rotation_gate(
                {i, i + 1}, {3, 3}, random.uniform());
        }
    }

    UINT param_count = circuit->get_parameter_count();
    for (UINT p = 0; p < param_count; ++p) {
        double current_angle = circuit->get_parameter(p);
        circuit->set_parameter(p, current_angle + random.uniform());
    }

    QuantumState state(n, true);
    circuit->update_quantum_state(&state);
    // std::cout << state << std::endl;
    // std::cout << circuit << std::endl;
    delete circuit;
}

TEST(ParametricCircuit, GateApplyDM) {
    const UINT n = 3;
    const UINT depth = 10;
    ParametricQuantumCircuit* circuit = new ParametricQuantumCircuit(n);
    Random random;
    for (UINT d = 0; d < depth; ++d) {
        for (UINT i = 0; i < n; ++i) {
            circuit->add_parametric_RX_gate(i, random.uniform());
            circuit->add_parametric_RY_gate(i, random.uniform());
            circuit->add_parametric_RZ_gate(i, random.uniform());
        }
        for (UINT i = d % 2; i + 1 < n; i += 2) {
            circuit->add_parametric_multi_Pauli_rotation_gate(
                {i, i + 1}, {3, 3}, random.uniform());
        }
    }

    UINT param_count = circuit->get_parameter_count();
    for (UINT p = 0; p < param_count; ++p) {
        double current_angle = circuit->get_parameter(p);
        circuit->set_parameter(p, current_angle + random.uniform());
    }

    DensityMatrix state(n);
    circuit->update_quantum_state(&state);
    // std::cout << state << std::endl;
    // std::cout << circuit << std::endl;
    delete circuit;
}

TEST(ParametricCircuit, ParametricGatePosition) {
    auto circuit = ParametricQuantumCircuit(3);
    circuit.add_parametric_RX_gate(0, 0.);
    circuit.add_H_gate(0);
    auto prz0 = gate::ParametricRZ(0, 0.);
    circuit.add_parametric_gate_copy(prz0);
    delete prz0;
    auto cz01 = gate::CNOT(0, 1);
    circuit.add_gate_copy(cz01);
    delete cz01;
    circuit.add_parametric_RY_gate(1, 0.);
    circuit.add_parametric_gate(gate::ParametricRY(2), 2);
    auto x0 = gate::X(0);
    circuit.add_gate_copy(x0, 2);
    delete x0;
    circuit.add_parametric_gate(gate::ParametricRZ(1), 0);
    circuit.remove_gate(4);
    circuit.remove_gate(5);
    auto ppr1 = gate::ParametricPauliRotation({1}, {0}, 0.);
    circuit.add_parametric_gate_copy(ppr1, 6);
    delete ppr1;

    ASSERT_EQ(circuit.get_parameter_count(), 5);
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 1);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 4);
    ASSERT_EQ(circuit.get_parametric_gate_position(2), 5);
    ASSERT_EQ(circuit.get_parametric_gate_position(3), 0);
    ASSERT_EQ(circuit.get_parametric_gate_position(4), 6);
}

// Regression test: moving a gate has to keep the registered parametric gate
// positions up to date.  Otherwise get_parametric_gate_position() silently
// reports the wrong gate, and every consumer of the positions (copy(),
// GradientByHalfPi, CausalConeSimulator) works on the wrong mapping.
TEST(ParametricCircuit, MoveGateKeepsParameterPositions) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 0
    circuit.add_X_gate(0);                   // gate 1
    circuit.add_parametric_RZ_gate(1, 0.7);  // parameter 1, gate 2
    circuit.add_H_gate(1);                   // gate 3

    circuit.move_gate(0, 3);
    // the moved parametric gate is at the back of the circuit, and the
    // parametric gate that was at gate 2 has been shifted to gate 1
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 3);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 1);
    ASSERT_TRUE(circuit.gate_list[3]->is_parametric());
    ASSERT_TRUE(circuit.gate_list[1]->is_parametric());
    ASSERT_NEAR(circuit.get_parameter(0), 0.3, eps);
    ASSERT_NEAR(circuit.get_parameter(1), 0.7, eps);

    // the moved circuit must be equivalent to a circuit built in that order
    ParametricQuantumCircuit expected(n);
    expected.add_X_gate(0);
    expected.add_parametric_RZ_gate(1, 0.);  // parameter 0 of expected
    expected.add_H_gate(1);
    expected.add_parametric_RX_gate(0, 0.);  // parameter 1 of expected
    const double rx_angle = 1.1;
    const double rz_angle = 2.2;
    circuit.set_parameter(0, rx_angle);   // parameter 0 is the RX gate
    circuit.set_parameter(1, rz_angle);   // parameter 1 is the RZ gate
    expected.set_parameter(1, rx_angle);  // parameter 1 of expected is RX
    expected.set_parameter(0, rz_angle);  // parameter 0 of expected is RZ

    QuantumState state(n), expected_state(n);
    state.set_Haar_random_state();
    expected_state.load(&state);
    circuit.update_quantum_state(&state);
    expected.update_quantum_state(&expected_state);
    ASSERT_STATE_NEAR(state, expected_state, eps);
}

TEST(ParametricCircuit, MoveGateNonParametricGateShiftsParameters) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 0
    circuit.add_X_gate(0);                   // gate 1
    circuit.add_parametric_RZ_gate(1, 0.7);  // parameter 1, gate 2

    circuit.move_gate(1, 2);  // the X gate moves behind the RZ gate
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 0);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 1);
    ASSERT_TRUE(circuit.gate_list[1]->is_parametric());

    ParametricQuantumCircuit expected(n);
    expected.add_parametric_RX_gate(0, 0.);
    expected.add_parametric_RZ_gate(1, 0.);
    expected.add_X_gate(0);
    const double rx_angle = 1.1;
    const double rz_angle = 2.2;
    circuit.set_parameter(0, rx_angle);
    circuit.set_parameter(1, rz_angle);
    expected.set_parameter(0, rx_angle);
    expected.set_parameter(1, rz_angle);

    QuantumState state(n), expected_state(n);
    state.set_Haar_random_state();
    expected_state.load(&state);
    circuit.update_quantum_state(&state);
    expected.update_quantum_state(&expected_state);
    ASSERT_STATE_NEAR(state, expected_state, eps);
}

TEST(ParametricCircuit, MoveGateBackwardKeepsParameterPositions) {
    const UINT n = 3;
    ParametricQuantumCircuit circuit(n);
    circuit.add_X_gate(0);                   // gate 0
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 1
    circuit.add_parametric_RZ_gate(1, 0.7);  // parameter 1, gate 2
    circuit.add_CNOT_gate(0, 1);             // gate 3
    circuit.add_parametric_RY_gate(2, 1.5);  // parameter 2, gate 4

    circuit.move_gate(4, 0);
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 2);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 3);
    ASSERT_EQ(circuit.get_parametric_gate_position(2), 0);
    for (UINT i = 0; i < circuit.get_parameter_count(); ++i) {
        ASSERT_TRUE(circuit.gate_list[circuit.get_parametric_gate_position(i)]
                        ->is_parametric());
    }

    ParametricQuantumCircuit expected(n);
    expected.add_parametric_RY_gate(2, 0.);  // parameter 0 of expected
    expected.add_X_gate(0);
    expected.add_parametric_RX_gate(0, 0.);  // parameter 1 of expected
    expected.add_parametric_RZ_gate(1, 0.);  // parameter 2 of expected
    expected.add_CNOT_gate(0, 1);
    const double rx_angle = 1.1;
    const double rz_angle = 2.2;
    const double ry_angle = 3.3;
    circuit.set_parameter(0, rx_angle);   // parameter 0 is the RX gate
    circuit.set_parameter(1, rz_angle);   // parameter 1 is the RZ gate
    circuit.set_parameter(2, ry_angle);   // parameter 2 is the RY gate
    expected.set_parameter(0, ry_angle);  // parameter 0 of expected is RY
    expected.set_parameter(1, rx_angle);  // parameter 1 of expected is RX
    expected.set_parameter(2, rz_angle);  // parameter 2 of expected is RZ

    QuantumState state(n), expected_state(n);
    state.set_Haar_random_state();
    expected_state.load(&state);
    circuit.update_quantum_state(&state);
    expected.update_quantum_state(&expected_state);
    ASSERT_STATE_NEAR(state, expected_state, eps);
}

// Regression test: the destination bound of the backward shift is inclusive.
// A parametric gate that already sits at to_index has to shift up when another
// gate is moved backward into to_index.  This distinguishes the correct
// `lower <= position` from the plausible wrong `lower < position`.
TEST(ParametricCircuit, MoveGateBackwardDestinationBoundary) {
    const UINT n = 1;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 0 (= to_index)
    circuit.add_X_gate(0);                   // gate 1
    circuit.add_parametric_RZ_gate(0, 0.7);  // parameter 1, gate 2

    circuit.move_gate(2, 0);

    ASSERT_EQ(circuit.get_parametric_gate_position(1), 0);  // moved gate
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 1);  // shifted up
    ASSERT_TRUE(circuit.gate_list[0]->is_parametric());
    ASSERT_TRUE(circuit.gate_list[1]->is_parametric());
    ASSERT_FALSE(circuit.gate_list[2]->is_parametric());
    ASSERT_NEAR(circuit.get_parameter(0), 0.3, eps);
    ASSERT_NEAR(circuit.get_parameter(1), 0.7, eps);
}

// Regression test: move_gate(i, i) is a no-op and must leave both the gate
// order and the registered parametric gate positions untouched.
TEST(ParametricCircuit, MoveGateSameIndexIsNoOp) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 0
    circuit.add_X_gate(0);                   // gate 1
    circuit.add_parametric_RZ_gate(1, 0.7);  // parameter 1, gate 2

    const std::vector<QuantumGateBase*> before = circuit.gate_list;

    circuit.move_gate(1, 1);  // non-parametric gate
    ASSERT_EQ(circuit.gate_list.size(), before.size());
    for (UINT i = 0; i < before.size(); ++i)
        ASSERT_EQ(circuit.gate_list[i], before[i]);
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 0);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 2);

    circuit.move_gate(2, 2);  // parametric gate
    ASSERT_EQ(circuit.gate_list.size(), before.size());
    for (UINT i = 0; i < before.size(); ++i)
        ASSERT_EQ(circuit.gate_list[i], before[i]);
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 0);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 2);
    ASSERT_NEAR(circuit.get_parameter(0), 0.3, eps);
    ASSERT_NEAR(circuit.get_parameter(1), 0.7, eps);
}

// Regression test: an out-of-range index throws and leaves the circuit
// unchanged.  The base implementation validates both indices before mutating,
// so the derived override must preserve that guarantee.  The invariant is
// re-checked after every individual call, not only once at the end.
TEST(ParametricCircuit, MoveGateInvalidIndexThrows) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);  // parameter 0, gate 0
    circuit.add_X_gate(0);                   // gate 1
    circuit.add_parametric_RZ_gate(1, 0.7);  // parameter 1, gate 2

    const std::vector<QuantumGateBase*> before = circuit.gate_list;
    auto check_unchanged = [&]() {
        ASSERT_EQ(circuit.gate_list.size(), before.size());
        for (UINT i = 0; i < before.size(); ++i)
            ASSERT_EQ(circuit.gate_list[i], before[i]);
        ASSERT_EQ(circuit.get_parametric_gate_position(0), 0);
        ASSERT_EQ(circuit.get_parametric_gate_position(1), 2);
        ASSERT_NEAR(circuit.get_parameter(0), 0.3, eps);
        ASSERT_NEAR(circuit.get_parameter(1), 0.7, eps);
    };

    ASSERT_THROW(circuit.move_gate(3, 0), GateIndexOutOfRangeException);
    check_unchanged();

    ASSERT_THROW(circuit.move_gate(0, 3), GateIndexOutOfRangeException);
    check_unchanged();

    ASSERT_THROW(circuit.move_gate(3, 3), GateIndexOutOfRangeException);
    check_unchanged();
}

class MyRandomCircuit : public ParametricCircuitBuilder {
    ParametricQuantumCircuit* create_circuit(
        UINT output_dim, UINT param_count) const override {
        ParametricQuantumCircuit* circuit =
            new ParametricQuantumCircuit(output_dim);
        UINT depth = param_count / output_dim;
        if (param_count % output_dim > 0) depth++;
        UINT param_index = 0;
        for (UINT d = 0; d < depth; ++d) {
            for (UINT i = 0; i < output_dim; ++i) {
                if (param_index < param_count) {
                    circuit->add_parametric_gate(gate::ParametricRX(i, 0.));
                    param_index++;
                } else {
                    circuit->add_gate(gate::RX(i, 0.0));
                }
            }
            for (UINT i = depth % 2; i + 1 < output_dim; ++i) {
                circuit->add_gate(gate::CNOT(0, 1));
            }
        }
        return circuit;
    }
};

TEST(EnergyMinimization, SingleQubitClassical) {
    const UINT n = 1;

    // define quantum circuit as prediction model
    std::function<ParametricQuantumCircuit*(UINT, UINT)> func =
        [](unsigned int qubit_count,
            unsigned int param_count) -> ParametricQuantumCircuit* {
        ParametricQuantumCircuit* circuit =
            new ParametricQuantumCircuit(qubit_count);
        for (unsigned int i = 0; i < qubit_count; ++i) {
            circuit->add_parametric_gate(gate::ParametricRX(i));
        }
        return circuit;
    };

    Observable* observable = new Observable(n);
    observable->add_operator(1.0, "Z 0");

    EnergyMinimizationProblem* emp = new EnergyMinimizationProblem(observable);

    QuantumCircuitEnergyMinimizationSolver qcems(&func, 0);
    qcems.solve(emp, 1000, "GD");
    double qc_loss = qcems.get_loss();

    DiagonalizationEnergyMinimizationSolver dems;
    dems.solve(emp);
    double diag_loss = dems.get_loss();

    EXPECT_NEAR(qc_loss, diag_loss, 1e-2);

    delete emp;
}

TEST(EnergyMinimization, SingleQubitComplex) {
    const UINT n = 1;

    // define quantum circuit as prediction model
    std::function<ParametricQuantumCircuit*(UINT, UINT)> func =
        [](unsigned int qubit_count,
            unsigned int param_count) -> ParametricQuantumCircuit* {
        ParametricQuantumCircuit* circuit =
            new ParametricQuantumCircuit(qubit_count);
        for (unsigned int i = 0; i < qubit_count; ++i) {
            circuit->add_parametric_gate(gate::ParametricRX(i));
            circuit->add_parametric_gate(gate::ParametricRY(i));
            circuit->add_parametric_gate(gate::ParametricRX(i));
        }
        return circuit;
    };

    Observable* observable = new Observable(n);
    observable->add_operator(1.0, "Z 0");
    observable->add_operator(1.0, "X 0");
    observable->add_operator(1.0, "Y 0");

    EnergyMinimizationProblem* emp = new EnergyMinimizationProblem(observable);

    QuantumCircuitEnergyMinimizationSolver qcems(&func, 0);
    qcems.solve(emp, 1000, "GD");
    double qc_loss = qcems.get_loss();

    DiagonalizationEnergyMinimizationSolver dems;
    dems.solve(emp);
    double diag_loss = dems.get_loss();

    EXPECT_NEAR(qc_loss, diag_loss, 1e-2);

    delete emp;
}

TEST(EnergyMinimization, MultiQubit) {
    const UINT n = 2;

    // define quantum circuit as prediction model
    std::function<ParametricQuantumCircuit*(UINT, UINT)> func =
        [](unsigned int qubit_count,
            unsigned int param_count) -> ParametricQuantumCircuit* {
        ParametricQuantumCircuit* circuit =
            new ParametricQuantumCircuit(qubit_count);
        for (unsigned int i = 0; i < qubit_count; ++i) {
            circuit->add_parametric_gate(gate::ParametricRX(i));
            circuit->add_parametric_gate(gate::ParametricRY(i));
            circuit->add_parametric_gate(gate::ParametricRX(i));
        }
        for (unsigned int i = 0; i + 1 < qubit_count; i += 2) {
            circuit->add_CNOT_gate(i, i + 1);
        }
        for (unsigned int i = 0; i < qubit_count; ++i) {
            circuit->add_parametric_gate(gate::ParametricRX(i));
            circuit->add_parametric_gate(gate::ParametricRY(i));
            circuit->add_parametric_gate(gate::ParametricRX(i));
        }
        return circuit;
    };

    Observable* observable = new Observable(n);
    observable->add_operator(1.0, "Z 0 X 1");
    observable->add_operator(-1.0, "Z 0 Y 1");
    observable->add_operator(0.2, "Y 0 Y 1");

    EnergyMinimizationProblem* emp = new EnergyMinimizationProblem(observable);

    QuantumCircuitEnergyMinimizationSolver qcems(&func, 0);
    qcems.solve(emp, 1000, "GD");
    double qc_loss = qcems.get_loss();

    DiagonalizationEnergyMinimizationSolver dems;
    dems.solve(emp);
    double diag_loss = dems.get_loss();
    // std::cout << qc_loss << " " << diag_loss << std::endl;
    ASSERT_GT(qc_loss, diag_loss);
    EXPECT_NEAR(qc_loss, diag_loss, 1e-1);

    delete emp;
}

TEST(ParametricGate, DuplicateIndex) {
    auto gate1 = gate::ParametricPauliRotation(
        {0, 1, 2, 3, 4, 5, 6}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
    EXPECT_TRUE(gate1 != NULL);
    delete gate1;
    auto gate2 = gate::ParametricPauliRotation(
        {2, 1, 0, 3, 7, 9, 4}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
    EXPECT_TRUE(gate2 != NULL);
    delete gate2;
    ASSERT_THROW(
        {
            auto gate3 = gate::ParametricPauliRotation(
                {0, 1, 3, 1, 5, 6, 2}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
        },
        DuplicatedQubitIndexException);
    ASSERT_THROW(
        {
            auto gate4 = gate::ParametricPauliRotation(
                {0, 3, 5, 2, 5, 6, 2}, {0, 0, 0, 0, 0, 0, 0}, 0.0);
        },
        DuplicatedQubitIndexException);
}

TEST(ParametricQuantumCircuitSimulator, Basic) {
    UINT n = 3;
    Observable observable(n);
    observable.add_operator(1., "Z 0");
    QuantumState state(n), test_state(n);
    ParametricQuantumCircuit circuit(n);
    for (UINT i = 0; i < n; ++i) {
        circuit.add_parametric_RX_gate(i, 1.0);
        circuit.add_parametric_RY_gate(i, 1.0);
    }
    ParametricQuantumCircuitSimulator sim(&circuit, &state);
    sim.simulate();
    // Circuitに適用した量子状態の期待値とSimulatorの期待値が同じであること
    circuit.update_quantum_state(&test_state);
    ASSERT_EQ(sim.get_expectation_value(&observable),
        observable.get_expectation_value(&test_state));
}

TEST(ParametricQuantumCircuitSimulator_multicpu, Basic) {
    UINT n = 3;
    Observable observable(n);
    observable.add_operator(1., "Z 0");
    QuantumState state(n, true), test_state(n, true);
    ParametricQuantumCircuit circuit(n);
    for (UINT i = 0; i < n; ++i) {
        circuit.add_parametric_RX_gate(i, 1.0);
        circuit.add_parametric_RY_gate(i, 1.0);
    }
    ParametricQuantumCircuitSimulator sim(&circuit, &state);
    sim.simulate();
    // Circuitに適用した量子状態の期待値とSimulatorの期待値が同じであること
    circuit.update_quantum_state(&test_state);
    ASSERT_NEAR(std::real(sim.get_expectation_value(&observable)),
        std::real(observable.get_expectation_value(&test_state)), 1e-12);
}

TEST(GradCalculator, BasicCheck) {
    Random rnd;
    unsigned int n = 5;
    Observable observable(n);
    std::string Pauli_string = "";
    for (int i = 0; i < n; ++i) {
        double coef = rnd.uniform();
        std::string Pauli_string = "Z ";
        Pauli_string += std::to_string(i);
        observable.add_operator(coef, Pauli_string.c_str());
    }

    ParametricQuantumCircuit circuit(n);
    for (int depth = 0; depth < 2; ++depth) {
        for (int i = 0; i < n; ++i) {
            circuit.add_parametric_RX_gate(i, 0);
            circuit.add_parametric_RZ_gate(i, 0);
        }

        for (int i = 0; i + 1 < n; i += 2) {
            circuit.add_CNOT_gate(i, i + 1);
        }

        for (int i = 1; i + 1 < n; i += 2) {
            circuit.add_CNOT_gate(i, i + 1);
        }
    }
    UINT parameter_count = circuit.get_parameter_count();
    std::vector<double> theta;
    for (int i = 0; i < parameter_count; ++i) {
        theta.push_back(rnd.uniform() * 5.0);
    }

    GradCalculator grad_calculator;
    auto grad_calculator_theta_specified_in_function_call_result =
        grad_calculator.calculate_grad(circuit, observable, theta);

    for (UINT i = 0; i < parameter_count; ++i) {
        ASSERT_EQ(circuit.get_parameter(i), 0);
        circuit.set_parameter(i, theta[i]);
    }
    auto grad_calculator_theta_in_circuit_result =
        grad_calculator.calculate_grad(circuit, observable);

    std::vector<std::complex<double>> naive_method_result(parameter_count);
    {
        const double delta = 0.001;
        for (int i = 0; i < parameter_count; ++i) {
            std::complex<double> plus_delta, minus_delta;
            {
                for (int q = 0; q < parameter_count; ++q) {
                    if (i == q) {
                        circuit.set_parameter(q, theta[q] + delta);
                    } else {
                        circuit.set_parameter(q, theta[q]);
                    }
                }
                CausalConeSimulator cone(circuit, observable);
                plus_delta = cone.get_expectation_value();
            }
            {
                for (int q = 0; q < parameter_count; ++q) {
                    if (i == q) {
                        circuit.set_parameter(q, theta[q] - delta);
                    } else {
                        circuit.set_parameter(q, theta[q]);
                    }
                }
                CausalConeSimulator cone(circuit, observable);
                minus_delta = cone.get_expectation_value();
            }
            naive_method_result[i] = (plus_delta - minus_delta) / (2.0 * delta);
        }
    }
    for (int i = 0; i < parameter_count; ++i) {
        ASSERT_LT(
            abs(grad_calculator_theta_specified_in_function_call_result[i] -
                naive_method_result[i]),
            1e-6);
        ASSERT_LT(abs(grad_calculator_theta_in_circuit_result[i] -
                      naive_method_result[i]),
            1e-6);
    }
}

TEST(ParametricCircuit, ParametricMergeCircuits) {
    ParametricQuantumCircuit base_circuit(3), circuit_for_merge(3),
        expected_circuit(3);
    Random random;

    for (int i = 0; i < 3; ++i) {
        double initial_angle = random.uniform();
        base_circuit.add_parametric_RX_gate(i, initial_angle);
        base_circuit.add_X_gate(i);
        expected_circuit.add_parametric_RX_gate(i, initial_angle);
        expected_circuit.add_X_gate(i);
    }

    for (int i = 0; i < 3; ++i) {
        double initial_angle = random.uniform();
        circuit_for_merge.add_parametric_RX_gate(i, initial_angle);
        circuit_for_merge.add_X_gate(i);
        expected_circuit.add_parametric_RX_gate(i, initial_angle);
        expected_circuit.add_X_gate(i);
    }

    base_circuit.merge_circuit(&circuit_for_merge);

    ASSERT_EQ(base_circuit.to_string(), expected_circuit.to_string());
    UINT parametric_gate_index = 0;
    for (int i = 0; i < base_circuit.gate_list.size(); ++i) {
        ASSERT_EQ(base_circuit.gate_list[i]->to_string(),
            expected_circuit.gate_list[i]->to_string());
        if (base_circuit.gate_list[i]->is_parametric()) {
            // Compare parametric_gate angles
            ASSERT_NEAR(base_circuit.get_parameter(parametric_gate_index),
                expected_circuit.get_parameter(parametric_gate_index), eps);
            ++parametric_gate_index;
        }
    }
}
