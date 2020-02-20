#include <cassert>
#include "gate.hpp"
#include "gate_matrix.hpp"
#include <algorithm>
#include <functional>
#include <sstream>

bool QuantumGateBase::is_commute(const QuantumGateBase* gate) const{
    for(auto val1 : this->_target_qubit_list){
        for(auto val2 : gate->_target_qubit_list){
            // check gate1 target qubit - gate2 target qubit are commuting
            if ( val1.is_commute_with(val2) == false) return false;
        }
        for(auto val2 : gate->_control_qubit_list){
            // check gate1 target qubit - gate2 control qubit are commuting
            if ( val1.is_commute_with(val2) == false) return false;
        }
    }
    for(auto val1 : this->_control_qubit_list){
        for(auto val2 : gate->_target_qubit_list){
            // check gate1 control qubit - gate2 target qubit are commuting
            if ( val1.is_commute_with(val2) == false) return false;
        }
    }
    // Note: gate1 control - gate2 control are always commuting
    return true;
}

bool QuantumGateBase::is_Pauli() const{
    return (this->_gate_property & FLAG_PAULI) != 0;
}
bool QuantumGateBase::is_Clifford() const{
    return (this->_gate_property & FLAG_CLIFFORD) != 0;
}
bool QuantumGateBase::is_Gaussian() const{
    return (this->_gate_property & FLAG_GAUSSIAN) != 0;
}
bool QuantumGateBase::is_parametric() const{
    return (this->_gate_property & FLAG_PARAMETRIC) != 0;
}

bool QuantumGateBase::is_diagonal() const{
    // if all the qubits commute with Z Pauli, the matrix of a gate is diagonal in Z basis
    for(auto val : this->_target_qubit_list){
        if( val.is_commute_Z() == false ){
            return false;
        }
    }
    return true;
}

UINT QuantumGateBase::get_property_value() const {
    return this->_gate_property;
}

bool QuantumGateBase::commute_Pauli_at(UINT qubit_index, UINT pauli_type) const{
    if(pauli_type == 0) return true;
    if(pauli_type >= 4) {
        fprintf(stderr,"invalid Pauli id is given\n");
        assert(false);
    }
    auto ite_target = std::find_if(this->_target_qubit_list.begin(), this->_target_qubit_list.end(), [&](QubitInfo v){ return v.index()==qubit_index;} );
    if(ite_target != this->_target_qubit_list.end()){
        switch(pauli_type){
        case 1: // X
            return ite_target->is_commute_X();
            break;
        case 2: // Y
            return ite_target->is_commute_Y();
            break;
        case 3: // Z
            return ite_target->is_commute_Z();
            break;
        }
    }

    auto ite_control = std::find_if(this->_control_qubit_list.begin(), this->_control_qubit_list.end(), [&](QubitInfo v){ return v.index()==qubit_index;} );
    if(ite_control != this->_control_qubit_list.end()){
        if(pauli_type == 3) return true;
        else return false;
    }
    return true;
}

std::string QuantumGateBase::to_string() const {
    std::stringstream stream;
    stream << " *** gate info *** " << std::endl;
    stream << " * gate name : " << this->_name << std::endl;
    stream << " * target    : " << std::endl;
    for (auto val : this->_target_qubit_list) {
        stream << " " << val.index() << " : commute "
            << (val.is_commute_X() ? "X" : " ") << " "
            << (val.is_commute_Y() ? "Y" : " ") << " "
            << (val.is_commute_Z() ? "Z" : " ") << " "
            << std::endl;
    }
    stream << " * control   : " << std::endl;
    for (auto val : this->_control_qubit_list) {
        stream << " " << val.index() << " : value " << val.control_value() << std::endl;
    }
    stream << " * Pauli     : " << (this->is_Pauli() ? "yes" : "no") << std::endl;
    stream << " * Clifford  : " << (this->is_Clifford() ? "yes" : "no") << std::endl;
    stream << " * Gaussian  : " << (this->is_Gaussian() ? "yes" : "no") << std::endl;
    stream << " * Parametric: " << (this->is_parametric() ? "yes" : "no") << std::endl;
    stream << " * Diagonal  : " << (this->is_diagonal() ? "yes" : "no") << std::endl;
    return stream.str();
}

std::string QuantumGateBase::get_name () const {
    return this->_name;
}

std::ostream& operator<<(std::ostream& stream, const QuantumGateBase& gate) {
    stream << gate.to_string();
    return stream;
}
std::ostream& operator<<(std::ostream& stream, const QuantumGateBase* gate) {
    stream << *gate;
    return stream;
}
