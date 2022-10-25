#include "gate_factory.cpp"
#include "gate_named_one.hpp"

QuantumGateBase* ClsOneQubitGate::inverse(void) {
    if (this->_name == "I") {
        return gate::Identity(this->target_qubit_list[0].index());
    }
    if (this->_name == "X") {
        return gate::X(this->target_qubit_list[0].index());
    }
    if (this->_name == "Y") {
        return gate::Y(this->target_qubit_list[0].index());
    }
    if (this->_name == "Z") {
        return gate::Z(this->target_qubit_list[0].index());
    }
    if (this->_name == "H") {
        return gate::H(this->target_qubit_list[0].index());
    }
    return this->QuantumGateBase::inverse();
}