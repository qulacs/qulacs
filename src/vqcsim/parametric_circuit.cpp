
#include <iostream>
#include "parametric_circuit.hpp"
#include "parametric_gate.hpp"
#include "parametric_gate_factory.hpp"

ParametricQuantumCircuit::ParametricQuantumCircuit(UINT qubit_count_) : QuantumCircuit(qubit_count_) {};
ParametricQuantumCircuit::ParametricQuantumCircuit(std::string qasm_path, std::string qasm_loader_script_path) : QuantumCircuit(qasm_path, qasm_loader_script_path) {
    // TODO: enables load of parametric gate
};


ParametricQuantumCircuit* ParametricQuantumCircuit::copy() const {
	ParametricQuantumCircuit* new_circuit = new ParametricQuantumCircuit(this->qubit_count);
	for (UINT gate_pos = 0; gate_pos < this->gate_list.size() ; gate_pos++){
		auto pos = std::find(this->_parametric_gate_position.begin(), this->_parametric_gate_position.end(), gate_pos);
		bool is_parametric = (pos != this->_parametric_gate_position.end());

		if (is_parametric) {
			new_circuit->add_parametric_gate((QuantumGate_SingleParameter*)this->gate_list[gate_pos]->copy());
		}
		else {
			new_circuit->add_gate(this->gate_list[gate_pos]->copy());
		}
	}
	return new_circuit;
}

void ParametricQuantumCircuit::add_parametric_gate(QuantumGate_SingleParameter* gate) {
    _parametric_gate_position.push_back((UINT)gate_list.size());
    this->add_gate(gate);
    _parametric_gate_list.push_back(gate);
};
void ParametricQuantumCircuit::add_parametric_gate(QuantumGate_SingleParameter* gate, UINT index) {
    _parametric_gate_position.push_back(index);
    this->add_gate(gate, index);
    _parametric_gate_list.push_back(gate);
}
UINT ParametricQuantumCircuit::get_parameter_count() const {
    return (UINT)_parametric_gate_list.size(); 
}
double ParametricQuantumCircuit::get_parameter(UINT index) const { 
	if (index >= this->_parametric_gate_list.size()) {
		std::cerr << "Error: ParametricQuantumCircuit::get_parameter(UINT): parameter index is out of range" << std::endl;
		return 0.;
	}

    return _parametric_gate_list[index]->get_parameter_value();
}
void ParametricQuantumCircuit::set_parameter(UINT index, double value) { 
	if (index >= this->_parametric_gate_list.size()) {
		std::cerr << "Error: ParametricQuantumCircuit::set_parameter(UINT,double): parameter index is out of range" << std::endl;
		return;
	}
	
	_parametric_gate_list[index]->set_parameter_value(value);
}


std::string ParametricQuantumCircuit::to_string() const {
    std::stringstream os;
	os << QuantumCircuit::to_string();
	os << "*** Parameter Info ***" << std::endl;
    os << "# of parameter: " << this->get_parameter_count() << std::endl;
    return os.str();
}


std::ostream& operator<<(std::ostream& stream, const ParametricQuantumCircuit& circuit) {
    stream << circuit.to_string();
    return stream;
}
std::ostream& operator<<(std::ostream& stream, const ParametricQuantumCircuit* gate) {
    stream << *gate;
    return stream;
}



UINT ParametricQuantumCircuit::get_parametric_gate_position(UINT index) const { 
	if (index >= this->_parametric_gate_list.size()) {
		std::cerr << "Error: ParametricQuantumCircuit::get_parametric_gate_position(UINT): parameter index is out of range" << std::endl;
		return 0;
	}
	
	return _parametric_gate_position[index];
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate) {
    QuantumCircuit::add_gate(gate);
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate, UINT index) {
    QuantumCircuit::add_gate(gate, index);
    for (auto& val : _parametric_gate_position) if (val >= index)val++;
}
void ParametricQuantumCircuit::add_gate_copy(const QuantumGateBase* gate) {
	QuantumCircuit::add_gate(gate->copy());
}
void ParametricQuantumCircuit::add_gate_copy(const QuantumGateBase* gate, UINT index) {
	QuantumCircuit::add_gate(gate->copy(), index);
	for (auto& val : _parametric_gate_position) if (val >= index)val++;
}

void ParametricQuantumCircuit::remove_gate(UINT index) {
    auto ite = std::find(_parametric_gate_position.begin(), _parametric_gate_position.end(), (unsigned int)index);
    if (ite != _parametric_gate_position.end()) {
        UINT dist = (UINT)std::distance(_parametric_gate_position.begin(), ite);
        _parametric_gate_position.erase(_parametric_gate_position.begin() + dist);
        _parametric_gate_list.erase(_parametric_gate_list.begin() + dist);
    }
    QuantumCircuit::remove_gate(index);
    for (auto& val : _parametric_gate_position) if (val >= index)val--;
}


void ParametricQuantumCircuit::add_parametric_RX_gate(UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRX(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RY_gate(UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRY(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RZ_gate(UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRZ(target_index, initial_angle));
}

void ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate(std::vector<UINT> target, std::vector<UINT> pauli_id, double initial_angle) {
    this->add_parametric_gate(gate::ParametricPauliRotation(target, pauli_id, initial_angle));
}
