#pragma once

#define _USE_MATH_DEFINES
#include <cmath>

#include <cppsim/simulator.hpp>
#include "parametric_circuit.hpp"
#include "parametric_simulator.hpp"
#include "problem.hpp"

class DllExport QuantumCircuitGradientDifferentiation {
public:
    virtual double compute_gradient(ParametricQuantumCircuitSimulator* sim, const EnergyMinimizationProblem* instance, const std::vector<double>& parameter, std::vector<double>* gradient) = 0;
};

class DllExport GradientByHalfPi : public QuantumCircuitGradientDifferentiation {
public:
    virtual double compute_gradient(ParametricQuantumCircuitSimulator* sim, const EnergyMinimizationProblem* instance, const std::vector<double>& parameter, std::vector<double>* gradient) override;
};
