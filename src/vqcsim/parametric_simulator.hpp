#pragma once

#include <cppsim/simulator.hpp>

class DllExport ParametricQuantumCircuitSimulator : public QuantumCircuitSimulator {
private:
    ParametricQuantumCircuit* _parametric_circuit;
public:
    ParametricQuantumCircuitSimulator(ParametricQuantumCircuit* circuit, QuantumStateBase* state = NULL);
    double get_parameter(UINT index) const;
    void add_parameter_value(UINT index, double value);
    void set_parameter_value(UINT index, double value);
    UINT get_parametric_gate_count();
    UINT get_parametric_gate_position(UINT index);
};
