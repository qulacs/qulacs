#pragma once
#include <cppsim/circuit_builder.hpp>

class ParametricCircuitBuilder : public QuantumCircuitBuilder {
    virtual ParametricQuantumCircuit* create_circuit(UINT output_dim, UINT param_count) const = 0;
};

