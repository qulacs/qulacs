#pragma once

#include <cppsim/type.hpp>
#include "parametric_gate.hpp"
#include <vector>
#include <string>

namespace gate {
    DllExport QuantumGateBase* create_parametric_quantum_gate_from_string(std::string gate_string);
    DllExport QuantumGate_SingleParameter* ParametricRX(UINT qubit_index, double initial_angle = 0.);
    DllExport QuantumGate_SingleParameter* ParametricRY(UINT qubit_index, double initial_angle = 0.);
    DllExport QuantumGate_SingleParameter* ParametricRZ(UINT qubit_index, double initial_angle = 0.);
    DllExport QuantumGate_SingleParameter* ParametricPauliRotation(std::vector<UINT> target, std::vector<UINT> pauli_id, double initial_angle = 0.);
}

