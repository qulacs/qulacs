#define _USE_MATH_DEFINES
#include "GradCalculator.hpp"

#include <math.h>

#include "causalcone_simulator.hpp"

std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& circuit, Observable& obs,
    std::vector<double> theta) {
    ParametricQuantumCircuit* circuit_copy = circuit.copy();
    UINT parameter_count = circuit_copy->get_parameter_count();

    std::vector<std::complex<double>> grad(parameter_count);
    for (UINT target_gate_itr = 0; target_gate_itr < parameter_count;
         target_gate_itr++) {
        std::complex<double> plus_delta, minus_delta;
        {
            for (UINT q = 0; q < parameter_count; ++q) {
                if (target_gate_itr == q) {
                    circuit_copy->set_parameter(q, theta[q] + M_PI_2);
                } else {
                    circuit_copy->set_parameter(q, theta[q]);
                }
            }
            CausalConeSimulator tmp(*circuit_copy, obs);
            plus_delta = tmp.get_expectation_value();
        }
        {
            for (UINT q = 0; q < parameter_count; ++q) {
                if (target_gate_itr == q) {
                    circuit_copy->set_parameter(q, theta[q] - M_PI_2);
                } else {
                    circuit_copy->set_parameter(q, theta[q]);
                }
            }
            CausalConeSimulator tmp(*circuit_copy, obs);
            minus_delta = tmp.get_expectation_value();
        }
        grad[target_gate_itr] = (plus_delta - minus_delta) / 2.0;
    }
    delete circuit_copy;
    return grad;
};

std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs) {
    std::vector<double> initial_parameter;
    for (UINT i = 0; i < x.get_parameter_count(); ++i) {
        initial_parameter.push_back(x.get_parameter(i));
    }
    return calculate_grad(x, obs, initial_parameter);
};
