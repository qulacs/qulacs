#include <gtest/gtest.h>

#include <complex>
#include <cppsim/gate_factory.hpp>
#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>
#include <vector>
#include <vqcsim/GradCalculator.hpp>
#include <vqcsim/causalcone_simulator.hpp>
#include <vqcsim/parametric_circuit.hpp>
#include <vqcsim/parametric_gate_factory.hpp>

#include "../util/util.hpp"

// Regression test: a copy must preserve the parameter index -> gate position
// mapping of the original circuit.  A parametric gate inserted in front of an
// already registered parametric gate makes that mapping non-monotonic, and
// registering parametric gates in gate order while copying silently renumbers
// the parameters of the copied circuit.
TEST(ParametricCircuit, CopyPreservesParameterOrder) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);                      // parameter 0
    circuit.add_parametric_gate(gate::ParametricRZ(1, 0.7), 0);  // parameter 1
    circuit.add_CNOT_gate(0, 1);

    ASSERT_EQ(circuit.get_parameter_count(), 2);
    ASSERT_EQ(circuit.get_parametric_gate_position(0), 1);
    ASSERT_EQ(circuit.get_parametric_gate_position(1), 0);

    ParametricQuantumCircuit* copy = circuit.copy();
    ASSERT_EQ(copy->get_parameter_count(), circuit.get_parameter_count());
    ASSERT_EQ(copy->gate_list.size(), circuit.gate_list.size());
    for (UINT i = 0; i < circuit.get_parameter_count(); ++i) {
        ASSERT_EQ(copy->get_parametric_gate_position(i),
            circuit.get_parametric_gate_position(i));
        ASSERT_NEAR(copy->get_parameter(i), circuit.get_parameter(i), eps);
        ASSERT_TRUE(copy->gate_list[copy->get_parametric_gate_position(i)]
                        ->is_parametric());
    }

    // the copy must stay equal to the original circuit when the same parameter
    // index is set on both of them
    QuantumState state(n), copy_state(n);
    state.set_Haar_random_state();
    copy_state.load(&state);
    for (UINT i = 0; i < circuit.get_parameter_count(); ++i) {
        circuit.set_parameter(i, 1.0 + i);
        copy->set_parameter(i, 1.0 + i);
    }
    circuit.update_quantum_state(&state);
    copy->update_quantum_state(&copy_state);
    ASSERT_STATE_NEAR(state, copy_state, eps);
    for (UINT i = 0; i < circuit.gate_list.size(); ++i) {
        ASSERT_EQ(
            copy->gate_list[i]->to_string(), circuit.gate_list[i]->to_string());
    }
    delete copy;
}

// Regression test: GradCalculator copies the circuit internally and then feeds
// the copy the parameter vector of the original circuit, so it silently
// returns the gradients of a different circuit when copy() renumbers the
// parameters.
TEST(ParametricCircuit, GradCalculatorWithPositionalInsert) {
    const UINT n = 2;
    ParametricQuantumCircuit circuit(n);
    circuit.add_parametric_RX_gate(0, 0.3);                      // parameter 0
    circuit.add_parametric_gate(gate::ParametricRZ(1, 0.7), 0);  // parameter 1
    circuit.add_CNOT_gate(0, 1);

    Observable observable(n);
    observable.add_operator(1.0, "Z 0");
    observable.add_operator(0.5, "X 1");

    const UINT parameter_count = circuit.get_parameter_count();
    std::vector<double> theta;
    for (UINT i = 0; i < parameter_count; ++i) {
        theta.push_back(circuit.get_parameter(i));
    }

    GradCalculator grad_calculator;
    std::vector<std::complex<double>> grad =
        grad_calculator.calculate_grad(circuit, observable);
    std::vector<double> backprop_grad = circuit.backprop(&observable);

    // reference gradient by central finite difference, using the parameter
    // indices of the circuit itself
    const double delta = 0.001;
    for (UINT i = 0; i < parameter_count; ++i) {
        std::complex<double> plus_delta, minus_delta;
        {
            for (UINT q = 0; q < parameter_count; ++q) {
                circuit.set_parameter(q, i == q ? theta[q] + delta : theta[q]);
            }
            CausalConeSimulator cone(circuit, observable);
            plus_delta = cone.get_expectation_value();
        }
        {
            for (UINT q = 0; q < parameter_count; ++q) {
                circuit.set_parameter(q, i == q ? theta[q] - delta : theta[q]);
            }
            CausalConeSimulator cone(circuit, observable);
            minus_delta = cone.get_expectation_value();
        }
        const double naive = (plus_delta - minus_delta).real() / (2.0 * delta);
        ASSERT_NEAR(grad[i].real(), naive, 1e-6);
        ASSERT_NEAR(backprop_grad[i], naive, 1e-6);
    }
}
