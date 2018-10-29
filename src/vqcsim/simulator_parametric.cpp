
#include "parametric_circuit.hpp"
#include "parametric_simulator.hpp"

ParametricQuantumCircuitSimulator::ParametricQuantumCircuitSimulator(ParametricQuantumCircuit* circuit, QuantumStateBase* state)
    : QuantumCircuitSimulator(circuit, state), _parametric_circuit(circuit) {}

double ParametricQuantumCircuitSimulator::get_parameter(UINT index) const {
    return _parametric_circuit->get_parameter(index);
}
void ParametricQuantumCircuitSimulator::add_parameter_value(UINT index, double value) {
    _parametric_circuit->set_parameter(index, _parametric_circuit->get_parameter(index) + value);
}
void ParametricQuantumCircuitSimulator::set_parameter_value(UINT index, double value) {
    _parametric_circuit->set_parameter(index, value);
}
UINT ParametricQuantumCircuitSimulator::get_parametric_gate_count() {
    return _parametric_circuit->get_parameter_count();
}
UINT ParametricQuantumCircuitSimulator::get_parametric_gate_position(UINT index) {
    return _parametric_circuit->get_parametric_gate_position(index);
}
