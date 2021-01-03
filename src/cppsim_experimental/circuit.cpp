#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "circuit.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <sstream>
#include <stdexcept>

#include "gate.hpp"
#include "gate_factory.hpp"
#include "gate_matrix.hpp"
#include "observable.hpp"
#include "pauli_operator.hpp"

bool check_gate_index(
    const QuantumCircuit* circuit, const QuantumGateBase* gate);

void QuantumCircuit::update_quantum_state(QuantumStateBase* state) {
    if (state->qubit_count != this->qubit_count) {
        std::cerr << "Error: "
                     "QuantumCircuit::update_quantum_state(QuantumStateBase) : "
                     "invalid qubit count"
                  << std::endl;
        return;
    }

    for (const auto& gate : this->_gate_list) {
        gate->update_quantum_state(state);
    }
}

void QuantumCircuit::update_quantum_state(
    QuantumStateBase* state, UINT start, UINT end) {
    if (state->qubit_count != this->qubit_count) {
        std::cerr
            << "Error: "
               "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
               "UINT) : invalid qubit count"
            << std::endl;
        return;
    }
    if (start > end) {
        std::cerr
            << "Error: "
               "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
               "UINT) : start must be smaller than or equal to end"
            << std::endl;
        return;
    }
    if (end > this->_gate_list.size()) {
        std::cerr
            << "Error: "
               "QuantumCircuit::update_quantum_state(QuantumStateBase,UINT,"
               "UINT) : end must be smaller than or equal to gate_count"
            << std::endl;
        return;
    }
    for (UINT cursor = start; cursor < end; ++cursor) {
        this->_gate_list[cursor]->update_quantum_state(state);
    }
}


bool check_gate_index(
    const QuantumCircuit* circuit, const QuantumGateBase* gate) {
    auto vec1 = gate->get_target_index_list();
    auto vec2 = gate->get_control_index_list();
    UINT val = 0;
    if (vec1.size() > 0) {
        val = std::max(val, *std::max_element(vec1.begin(), vec1.end()));
    }
    if (vec2.size() > 0) {
        val = std::max(val, *std::max_element(vec2.begin(), vec2.end()));
    }
    return val < circuit->qubit_count;
}


void QuantumCircuit::add_noise_gate(
    QuantumGateBase* gate, std::string noise_type, double noise_prob) {

}

void QuantumCircuit::add_noise_gate_copy(
    QuantumGateBase* gate, std::string noise_type, double noise_prob) {
    this->add_noise_gate(gate->copy(), noise_type, noise_prob);
}
