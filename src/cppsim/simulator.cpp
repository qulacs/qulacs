#include "simulator.hpp"

#include <stdio.h>

#include "circuit.hpp"
#include "observable.hpp"
#include "state.hpp"

QuantumCircuitSimulator::QuantumCircuitSimulator(
    QuantumCircuit* circuit, QuantumStateBase* initial_state)
    : _circuit(circuit), _state(initial_state), _buffer(NULL) {
    if (initial_state == NULL) {
        _state = new QuantumState(this->_circuit->qubit_count);
        _own_state = true;
    }
    // _circuitはQuantumCircuitSimulatorを継承しているParametricQuantumCircuitSimulatorが
    // circuitのポインタを共有しておりtestで最適化した回路を使用するので、
    // 渡されたcircuitをそのまま使用する必要がある。
};

QuantumCircuitSimulator::~QuantumCircuitSimulator() {
    if (_own_state) {
        delete _state;
    }
    if (_buffer != NULL) {
        delete _buffer;
    }
}

void QuantumCircuitSimulator::initialize_state(ITYPE computational_basis) {
    _state->set_computational_basis(computational_basis);
}

void QuantumCircuitSimulator::initialize_random_state() {
    _state->set_Haar_random_state();
}

void QuantumCircuitSimulator::initialize_random_state(UINT seed) {
    _state->set_Haar_random_state(seed);
}

void QuantumCircuitSimulator::simulate() {
    _circuit->update_quantum_state(_state);
}
void QuantumCircuitSimulator::simulate_range(UINT start, UINT end) {
    _circuit->update_quantum_state(_state, start, end);
}

CPPCTYPE QuantumCircuitSimulator::get_expectation_value(
    const Observable* observable) {
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
    auto tmp = new QuantumState(_state->qubit_count);
    tmp->load(_buffer);
    _buffer->load(_state);
    _state->load(tmp);

    delete tmp;
}
