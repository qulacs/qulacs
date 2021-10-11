#pragma once
#include "parametric_circuit.hpp"

#include <iostream>

#include "cppsim/gate_factory.hpp"
#include "cppsim/gate_matrix.hpp"
#include "cppsim/gate_merge.hpp"
#include "cppsim/state.hpp"
#include "cppsim/type.hpp"
#include "parametric_gate.hpp"
#include "parametric_gate_factory.hpp"

ParametricQuantumCircuit::ParametricQuantumCircuit(UINT qubit_count_)
    : QuantumCircuit(qubit_count_){};

ParametricQuantumCircuit* ParametricQuantumCircuit::copy() const {
    ParametricQuantumCircuit* new_circuit =
        new ParametricQuantumCircuit(this->qubit_count);
    for (UINT gate_pos = 0; gate_pos < this->gate_list.size(); gate_pos++) {
        auto pos = std::find(this->_parametric_gate_position.begin(),
            this->_parametric_gate_position.end(), gate_pos);
        bool is_parametric = (pos != this->_parametric_gate_position.end());

        if (is_parametric) {
            new_circuit->add_parametric_gate(
                (QuantumGate_SingleParameter*)this->gate_list[gate_pos]
                    ->copy());
        } else {
            new_circuit->add_gate(this->gate_list[gate_pos]->copy());
        }
    }
    return new_circuit;
}

void ParametricQuantumCircuit::add_parametric_gate(
    QuantumGate_SingleParameter* gate) {
    _parametric_gate_position.push_back((UINT)gate_list.size());
    this->add_gate(gate);
    _parametric_gate_list.push_back(gate);
};
void ParametricQuantumCircuit::add_parametric_gate(
    QuantumGate_SingleParameter* gate, UINT index) {
    _parametric_gate_position.push_back(index);
    this->add_gate(gate, index);
    _parametric_gate_list.push_back(gate);
}
void ParametricQuantumCircuit::add_parametric_gate_copy(
    QuantumGate_SingleParameter* gate) {
    _parametric_gate_position.push_back((UINT)gate_list.size());
    QuantumGate_SingleParameter* copied_gate = gate->copy();
    QuantumCircuit::add_gate(copied_gate);
    _parametric_gate_list.push_back(copied_gate);
};
void ParametricQuantumCircuit::add_parametric_gate_copy(
    QuantumGate_SingleParameter* gate, UINT index) {
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
    _parametric_gate_position.push_back(index);
    QuantumGate_SingleParameter* copied_gate = gate->copy();
    QuantumCircuit::add_gate(copied_gate, index);
    _parametric_gate_list.push_back(copied_gate);
}
UINT ParametricQuantumCircuit::get_parameter_count() const {
    return (UINT)_parametric_gate_list.size();
}
double ParametricQuantumCircuit::get_parameter(UINT index) const {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr << "Error: ParametricQuantumCircuit::get_parameter(UINT): "
                     "parameter index is out of range"
                  << std::endl;
        return 0.;
    }

    return _parametric_gate_list[index]->get_parameter_value();
}
void ParametricQuantumCircuit::set_parameter(UINT index, double value) {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr
            << "Error: ParametricQuantumCircuit::set_parameter(UINT,double): "
               "parameter index is out of range"
            << std::endl;
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

std::ostream& operator<<(
    std::ostream& stream, const ParametricQuantumCircuit& circuit) {
    stream << circuit.to_string();
    return stream;
}
std::ostream& operator<<(
    std::ostream& stream, const ParametricQuantumCircuit* gate) {
    stream << *gate;
    return stream;
}

UINT ParametricQuantumCircuit::get_parametric_gate_position(UINT index) const {
    if (index >= this->_parametric_gate_list.size()) {
        std::cerr
            << "Error: "
               "ParametricQuantumCircuit::get_parametric_gate_position(UINT):"
               " parameter index is out of range"
            << std::endl;
        return 0;
    }

    return _parametric_gate_position[index];
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate) {
    QuantumCircuit::add_gate(gate);
}
void ParametricQuantumCircuit::add_gate(QuantumGateBase* gate, UINT index) {
    QuantumCircuit::add_gate(gate, index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
}
void ParametricQuantumCircuit::add_gate_copy(const QuantumGateBase* gate) {
    QuantumCircuit::add_gate(gate->copy());
}
void ParametricQuantumCircuit::add_gate_copy(
    const QuantumGateBase* gate, UINT index) {
    QuantumCircuit::add_gate(gate->copy(), index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val++;
}

void ParametricQuantumCircuit::remove_gate(UINT index) {
    auto ite = std::find(_parametric_gate_position.begin(),
        _parametric_gate_position.end(), (unsigned int)index);
    if (ite != _parametric_gate_position.end()) {
        UINT dist = (UINT)std::distance(_parametric_gate_position.begin(), ite);
        _parametric_gate_position.erase(
            _parametric_gate_position.begin() + dist);
        _parametric_gate_list.erase(_parametric_gate_list.begin() + dist);
    }
    QuantumCircuit::remove_gate(index);
    for (auto& val : _parametric_gate_position)
        if (val >= index) val--;
}

void ParametricQuantumCircuit::add_parametric_RX_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRX(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RY_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRY(target_index, initial_angle));
}
void ParametricQuantumCircuit::add_parametric_RZ_gate(
    UINT target_index, double initial_angle) {
    this->add_parametric_gate(gate::ParametricRZ(target_index, initial_angle));
}

void ParametricQuantumCircuit::add_parametric_multi_Pauli_rotation_gate(
    std::vector<UINT> target, std::vector<UINT> pauli_id,
    double initial_angle) {
    this->add_parametric_gate(
        gate::ParametricPauliRotation(target, pauli_id, initial_angle));
}

// watle made
using namespace std;
std::vector<double> ParametricQuantumCircuit::backprop(
    GeneralQuantumOperator* obs) {
    int n = this->qubit_count;
    QuantumState* state = new QuantumState(n);
    state->set_zero_state();
    this->update_quantum_state(state);
    // parametric bibunti tasu
    std::vector<CPPCTYPE> bibun(1 << (this->qubit_count));
    QuantumState* bistate = new QuantumState(n);
    QuantumState* Astate = new QuantumState(n);
    // cerr<<state<<endl;
    /*for(int i=0;i<target_qubit_index_list.size();i++){
        int tqi=target_qubit_index_list[i];

        Astate->load(state);
        if(target_qubit_pauli_list[i]==1){
            //pauli X
            auto x_gate=gate::X(tqi);
            x_gate->update_quantum_state(Astate);
            Astate->multiply_coef(-target_qubit_coef_list[i]);
            bistate->add_state(Astate);
            delete x_gate;
        }
        if(target_qubit_pauli_list[i]==2){
            //pauli Y
            auto y_gate=gate::Y(tqi);
            y_gate->update_quantum_state(Astate);
            Astate->multiply_coef(-target_qubit_coef_list[i]);
            bistate->add_state(Astate);
            delete y_gate;

        }
        if(target_qubit_pauli_list[i]==3){
            //pauli Z
            auto z_gate=gate::Z(tqi);
            z_gate->update_quantum_state(Astate);
            Astate->multiply_coef(-target_qubit_coef_list[i]);
            bistate->add_state(Astate);
            delete z_gate;
        }
    }*/
    bistate->load(state);

    obs->update_quantum_state(bistate);
    bistate->multiply_coef(-1);
    // cerr<<bistate<<endl;
    double ansnorm = bistate->get_squared_norm();
    // TODO: Not compare with 0, check if `ansnorm` is close enough to 0; e.g. `ansnorm < 1e-6`.
    if (ansnorm == 0) {
        vector<double> ans(this->get_parameter_count());
        return ans;
    }
    bistate->normalize(ansnorm);
    ansnorm = sqrt(ansnorm);
    int m = this->gate_list.size();
    vector<int> gyapgp(m, -1);  // prametric gate position no gyaku
    for (UINT i = 0; i < this->get_parameter_count(); i++) {
        gyapgp[this->_parametric_gate_position[i]] = i;
    }
    vector<double> ans(this->get_parameter_count());
    for (int i = m - 1; i >= 0; i--) {
        auto gate = (this->gate_list[i])->copy();
        if (gyapgp[i] != -1) {
            Astate->load(bistate);
            if (gate->get_name() != "ParametricRX" &&
                gate->get_name() != "ParametricRY" &&
                gate->get_name() != "ParametricRZ") {
                std::cerr << "Error: " << gate->get_name()
                          << " does not support backprop in parametric"
                          << std::endl;
            } else {
                double kaku = this->get_parameter(gyapgp[i]);
                this->set_parameter(gyapgp[i], 3.14159265358979);
                auto Dgate = (this->gate_list[i])->copy();
                Dgate->update_quantum_state(Astate);
                ans[gyapgp[i]] =
                    (state::inner_product(state, Astate) * ansnorm).real();
                this->set_parameter(gyapgp[i], kaku);
            }
        }

        auto Agate = gate::get_adjoint_gate(gate);
        Agate->update_quantum_state(bistate);
        Agate->update_quantum_state(state);
        delete Agate;
        delete gate;
    }
    delete Astate;
    delete state;
    delete bistate;

    return ans;
    // CPP
}
