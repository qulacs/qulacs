
#pragma once

#include <Eigen/Dense>
#include <cppsim/circuit_builder.hpp>
#include <cppsim/simulator.hpp>
#include <cppsim/type.hpp>
#include <cppsim/utility.hpp>
#include <functional>
#include <vector>

#include "differential.hpp"
#include "optimizer.hpp"
#include "problem.hpp"

class QuantumCircuitEnergyMinimizationSolver {
private:
    ParametricQuantumCircuit* _circuit;
    const std::function<ParametricQuantumCircuit*(UINT, UINT)>*
        _circuit_construction;
    UINT _param_count;
    std::vector<double> _parameter;
    double loss;

public:
    bool verbose;
    QuantumCircuitEnergyMinimizationSolver(
        const std::function<ParametricQuantumCircuit*(UINT, UINT)>*
            circuit_generator,
        UINT param_count = 0);

    virtual ~QuantumCircuitEnergyMinimizationSolver();

    virtual void solve(EnergyMinimizationProblem* instance,
        UINT max_iteration = 100, std::string optimizer_name = "GD",
        std::string differentiation_method = "HalfPi");

    virtual double get_loss();

    virtual std::vector<double> get_parameter();

    ParametricQuantumCircuitSimulator* get_quantum_circuit_simulator();
};

class DiagonalizationEnergyMinimizationSolver {
private:
    ParametricQuantumCircuit* _circuit;
    const std::function<ParametricQuantumCircuit*(UINT, UINT)>*
        _circuit_construction;
    UINT _param_count;
    std::vector<double> _parameter;
    double loss;

public:
    bool verbose;
    DiagonalizationEnergyMinimizationSolver();

    virtual ~DiagonalizationEnergyMinimizationSolver();

    virtual void solve(EnergyMinimizationProblem* instance);

    virtual double get_loss();
};
