#pragma once

#include <cppsim/observable.hpp>
#include <cppsim/state.hpp>

#include "parametric_circuit.hpp"

class DllExport GradCalculator {
public:
    std::vector<std::complex<double>> calculate_grad(
        ParametricQuantumCircuit& x, Observable& obs,
        std::vector<double> theta);
    std::vector<std::complex<double>> calculate_grad(
        ParametricQuantumCircuit& x, Observable& obs);
};
