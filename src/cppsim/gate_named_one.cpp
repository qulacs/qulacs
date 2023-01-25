#include "gate_factory.hpp"

ClsOneQubitGate* ClsOneQubitGate::get_inverse(void) const {
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
    if (this->_name == "S") {
        return gate::Sdag(this->target_qubit_list[0].index());
    }
    if (this->_name == "Sdag") {
        return gate::S(this->target_qubit_list[0].index());
    }
    if (this->_name == "T") {
        return gate::Tdag(this->target_qubit_list[0].index());
    }
    if (this->_name == "Tdag") {
        return gate::T(this->target_qubit_list[0].index());
    }
    if (this->_name == "sqrtX") {
        return gate::sqrtXdag(this->target_qubit_list[0].index());
    }
    if (this->_name == "sqrtXdag") {
        return gate::sqrtX(this->target_qubit_list[0].index());
    }
    if (this->_name == "sqrtY") {
        return gate::sqrtYdag(this->target_qubit_list[0].index());
    }
    if (this->_name == "sqrtYdag") {
        return gate::sqrtY(this->target_qubit_list[0].index());
    }
    if (this->_name == "Projection-0" || this->_name == "Projection-1") {
        throw NotImplementedException("Projection gate hasn't inverse gate");
    }
    throw NotImplementedException(
        "Inverse of " + this->_name + " gate is not Implemented");

    // return this->QuantumGateBase::get_inverse();
}
ClsOneQubitRotationGate* ClsOneQubitRotationGate::get_inverse(void) const {
    if (this->_name == "X-rotation") {
        return gate::RX(this->target_qubit_list[0].index(), -this->_angle);
    }
    if (this->_name == "Y-rotation") {
        return gate::RY(this->target_qubit_list[0].index(), -this->_angle);
    }
    if (this->_name == "Z-rotation") {
        return gate::RZ(this->target_qubit_list[0].index(), -this->_angle);
    }
    throw NotImplementedException(
        "Inverse of " + this->_name + " gate is not Implemented");
}