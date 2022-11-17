#include <stdio.h>
#include "simulator.hpp"
#include "state.hpp"
#include "observable.hpp"
#include "circuit.hpp"

QuantumCircuitSimulator::QuantumCircuitSimulator(QuantumCircuit* circuit, QuantumStateBase* initial_state)
    : _circuit(circuit), _state(initial_state), _buffer(NULL) {
    if (_state == NULL) _state = new QuantumState(this->_circuit->qubit_count);
};

QuantumCircuitSimulator::~QuantumCircuitSimulator() {
    if (_circuit != NULL) delete _circuit;
    if (_state != NULL)delete _state;
    if (_buffer != NULL) delete _buffer;
}

void QuantumCircuitSimulator::initialize_state(ITYPE computational_basis) {
    _state->set_computational_basis(computational_basis);
}

void QuantumCircuitSimulator::initialize_random_state() {
    _state->set_Haar_random_state();
}

void QuantumCircuitSimulator::simulate() {
    _circuit->update_quantum_state(_state);
}
void QuantumCircuitSimulator::simulate_range(UINT start, UINT end) {
    _circuit->update_quantum_state(_state, start, end);
}

CPPCTYPE QuantumCircuitSimulator::get_expectation_value(const Observable* observable) {
    return observable->get_expectation_value(_state);
}

UINT QuantumCircuitSimulator::get_gate_count() { 
    return (UINT)(_circuit->gate_list.size());
}
void QuantumCircuitSimulator::copy_state_to_buffer() {
    if (_buffer == NULL) _buffer = new QuantumState(_state->qubit_count);
    _buffer->load(_state);
}
void QuantumCircuitSimulator::copy_state_from_buffer() {
    if (_buffer == NULL) {
        _buffer = new QuantumState(_state->qubit_count);
        _buffer->set_zero_state();
    }
    _state->load(_buffer);
}
void QuantumCircuitSimulator::swap_state_and_buffer() {
    if (_buffer == NULL) {
        _buffer = new QuantumState(_state->qubit_count);
        _buffer->set_zero_state();
    }
    std::swap(_state, _buffer);
}


