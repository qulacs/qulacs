
#pragma once

#ifndef _MSC_VER
extern "C" {
#include <csim/update_ops.h>
}
#else
#include <csim/update_ops.h>
#endif

#include <cppsim/gate.hpp>
#include <cppsim/state.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/utility.hpp>

#ifndef _MSC_VER
#include <complex.h>
#endif


class QuantumGate_SingleParameter : public QuantumGateBase {
protected:
    double _angle;
    UINT _parameter_type;
public:
    QuantumGate_SingleParameter(double angle): _angle(angle) {
        _gate_property ^= FLAG_PARAMETRIC;
        _parameter_type = 0;
    }
    virtual void set_parameter_value(double value) { _angle = value; }
    virtual double get_parameter_value() const { return _angle; }
};

class QuantumGate_SingleParameterOneQubitRotation : public QuantumGate_SingleParameter {
protected:
    typedef void (T_UPDATE_FUNC)(UINT, double, CTYPE*, ITYPE);
    T_UPDATE_FUNC* _update_func;
    ComplexMatrix _matrix_element;

    QuantumGate_SingleParameterOneQubitRotation(double angle) : QuantumGate_SingleParameter(angle) {
        _angle = angle;
    };
public:
    virtual void update_quantum_state(QuantumStateBase* state) override {
        _update_func(this->_target_qubit_list[0].index(), _angle, state->data_c(), state->dim);
    };
    virtual QuantumGateBase* copy() const override {
        return new QuantumGate_SingleParameterOneQubitRotation(*this);
    };
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }
};


class ClsParametricRXGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRXGate(UINT target_qubit_index, double angle) : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_update_func = RX_gate;
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle), sin(_angle) * 1.i, sin(_angle) * 1.i, cos(_angle);
    }
};

class ClsParametricRYGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRYGate(UINT target_qubit_index, double angle) : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_update_func = RY_gate;
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle), sin(_angle), -sin(_angle), cos(_angle);
    }
};

class ClsParametricRZGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRZGate(UINT target_qubit_index, double angle) : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_update_func = RZ_gate;
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle) + 1.i*sin(_angle), 0, 0, cos(_angle) - 1.i * sin(_angle);
    }
};


class ClsParametricPauliRotationGate : public QuantumGate_SingleParameter {
protected:
    PauliOperator* _pauli;
public:
    ClsParametricPauliRotationGate(double angle, PauliOperator* pauli)
        : QuantumGate_SingleParameter(angle) {
        _pauli = pauli;
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        for (UINT index = 0; index < target_index_list.size(); ++index) {
            UINT commutation_relation = 0;
            if (pauli_id_list[index] == 1) commutation_relation = FLAG_X_COMMUTE;
            else if (pauli_id_list[index] == 2)commutation_relation = FLAG_Y_COMMUTE;
            else if (pauli_id_list[index] == 3)commutation_relation = FLAG_Z_COMMUTE;
            this->_target_qubit_list.push_back(TargetQubitInfo(target_index_list[index], commutation_relation));
        }
    };
    virtual ~ClsParametricPauliRotationGate() {
        delete _pauli;
    }
    virtual void update_quantum_state(QuantumStateBase* state) override {
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        multi_qubit_Pauli_rotation_gate_partial_list(
            target_index_list.data(), pauli_id_list.data(), (UINT)target_index_list.size(),
            _angle, state->data_c(), state->dim);
    };
    virtual QuantumGateBase* copy() const override {
        return new ClsParametricPauliRotationGate(_angle, _pauli->copy());
    };
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        get_Pauli_matrix(matrix, _pauli->get_pauli_id_list());
        matrix = cos(_angle)*ComplexMatrix::Identity(matrix.rows(), matrix.cols()) + 1.i * sin(_angle) * matrix;
    }
};

