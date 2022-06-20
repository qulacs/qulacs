#define _USE_MATH_DEFINES
#include "GradCalculator.hpp"

#include <math.h>

#include "causalcone_simulator.hpp"

std::vector<std::complex<double>> GradCalculator::calculate_grad(
    ParametricQuantumCircuit& x, Observable& obs, std::vector<double> theta) {
    ParametricQuantumCircuit* x_copy = x.copy();

    std::vector<std::complex<double>> grad;
    for (UINT i = 0; i < x.get_parameter_count(); ++i) {
        std::complex<double> y, z;
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                double diff = 0;
                if (i == q) {
                    diff = M_PI_2;
                }
                x_copy->set_parameter(q, theta[q] + diff);
            }
            CausalConeSimulator hoge(*x_copy, obs);
            y = hoge.get_expectation_value();
        }
        {
            for (UINT q = 0; q < x.get_parameter_count(); ++q) {
                double diff = 0;
                if (i == q) {
                    diff = M_PI_2;
                }
                x_copy->set_parameter(q, theta[q] - diff);
            }
            CausalConeSimulator hoge(*x_copy, obs);
            z = hoge.get_expectation_value();
        }
        grad.push_back((y - z) / 2.0);
    }
    delete x_copy;
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
