
#pragma once

#include <cppsim/exception.hpp>
#include <cppsim/gate.hpp>
#include <cppsim/pauli_operator.hpp>
#include <cppsim/state.hpp>
#include <cppsim/utility.hpp>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

class QuantumGate_SingleParameter : public QuantumGateBase {
protected:
    double _angle;
    UINT _parameter_type;

public:
    QuantumGate_SingleParameter(double angle) : _angle(angle) {
        _gate_property |= FLAG_PARAMETRIC;
        _parameter_type = 0;
    }
    virtual void set_parameter_value(double value) { _angle = value; }
    virtual double get_parameter_value() const { return _angle; }
    virtual QuantumGate_SingleParameter* copy() const override = 0;
};

class QuantumGate_SingleParameterOneQubitRotation
    : public QuantumGate_SingleParameter {
protected:
    using UpdateFunc = void (*)(UINT, double, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, double, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, double, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func = nullptr;
    UpdateFunc _update_func_dm = nullptr;
    UpdateFuncGpu _update_func_gpu = nullptr;
    UpdateFuncMpi _update_func_mpi = nullptr;

    QuantumGate_SingleParameterOneQubitRotation(double angle)
        : QuantumGate_SingleParameter(angle) {}

public:
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                if (_update_func_gpu == NULL) {
                    throw UndefinedUpdateFuncException(
                        "Error: "
                        "QuantumGate_SingleParameterOneQubitRotation::update_"
                        "quantum_state(QuantumStateBase) : update function "
                        "for GPU is undefined");
                }
                _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
                return;
            }
#endif
#ifdef _USE_MPI
            if (state->outer_qc > 0) {
                if (_update_func_mpi == NULL) {
                    throw UndefinedUpdateFuncException(
                        "Error: "
                        "QuantumGate_SingleParameterOneQubitRotation::update_"
                        "quantum_state(QuantumStateBase) : update function "
                        "for multi-cpu is undefined");
                }
                _update_func_mpi(this->_target_qubit_list[0].index(), _angle,
                    state->data_c(), state->dim, state->inner_qc);
                return;
            }
#endif
            if (_update_func == NULL) {
                throw UndefinedUpdateFuncException(
                    "Error: "
                    "QuantumGate_SingleParameterOneQubitRotation::update_"
                    "quantum_state(QuantumStateBase) : update function is "
                    "undefined");
            }
            {
                _update_func(this->_target_qubit_list[0].index(), _angle,
                    state->data_c(), state->dim);
            }
        } else {
            if (_update_func_dm == NULL) {
                throw UndefinedUpdateFuncException(
                    "Error: "
                    "QuantumGate_SingleParameterOneQubitRotation::update_"
                    "quantum_state(QuantumStateBase) : update function is "
                    "undefined");
            }
            _update_func_dm(this->_target_qubit_list[0].index(), _angle,
                state->data_c(), state->dim);
        }
    }
};

class ClsParametricRXGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRXGate(UINT target_qubit_index, double angle)
        : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_name = "ParametricRX";
        this->_update_func = RX_gate;
        this->_update_func_dm = dm_RX_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RX_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RX_gate_mpi;
#endif
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
    }
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = ComplexMatrix::Zero(2, 2);
        matrix << cos(_angle / 2), sin(_angle / 2) * 1.i, sin(_angle / 2) * 1.i,
            cos(_angle / 2);
    }
    virtual ClsParametricRXGate* copy() const override {
        return new ClsParametricRXGate(*this);
    };
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "ParametricRXGate");
        pt.put("target_qubit", _target_qubit_list[0].index());
        pt.put("angle", _angle);
        return pt;
    }
    virtual ClsParametricRXGate* get_inverse() const override {
        return new ClsParametricRXGate(
            this->target_qubit_list[0].index(), -_angle);
    };
};

class ClsParametricRYGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRYGate(UINT target_qubit_index, double angle)
        : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_name = "ParametricRY";
        this->_update_func = RY_gate;
        this->_update_func_dm = dm_RY_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RY_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RY_gate_mpi;
#endif
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
    }
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = ComplexMatrix::Zero(2, 2);
        matrix << cos(_angle / 2), sin(_angle / 2), -sin(_angle / 2),
            cos(_angle / 2);
    }
    virtual ClsParametricRYGate* copy() const override {
        return new ClsParametricRYGate(*this);
    };
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "ParametricRYGate");
        pt.put("target_qubit", _target_qubit_list[0].index());
        pt.put("angle", _angle);
        return pt;
    }
    virtual ClsParametricRYGate* get_inverse() const override {
        return new ClsParametricRYGate(
            this->target_qubit_list[0].index(), -_angle);
    };
};

class ClsParametricRZGate : public QuantumGate_SingleParameterOneQubitRotation {
public:
    ClsParametricRZGate(UINT target_qubit_index, double angle)
        : QuantumGate_SingleParameterOneQubitRotation(angle) {
        this->_name = "ParametricRZ";
        this->_update_func = RZ_gate;
        this->_update_func_dm = dm_RZ_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RZ_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RZ_gate_mpi;
#endif
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
    }
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = ComplexMatrix::Zero(2, 2);
        matrix << cos(_angle / 2) + 1.i * sin(_angle / 2), 0, 0,
            cos(_angle / 2) - 1.i * sin(_angle / 2);
    }
    virtual ClsParametricRZGate* copy() const override {
        return new ClsParametricRZGate(*this);
    };
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "ParametricRZGate");
        pt.put("target_qubit", _target_qubit_list[0].index());
        pt.put("angle", _angle);
        return pt;
    }
    virtual ClsParametricRZGate* get_inverse() const override {
        return new ClsParametricRZGate(
            this->target_qubit_list[0].index(), -_angle);
    };
};

class ClsParametricPauliRotationGate : public QuantumGate_SingleParameter {
protected:
    PauliOperator* _pauli;

public:
    ClsParametricPauliRotationGate(double angle, PauliOperator* pauli)
        : QuantumGate_SingleParameter(angle) {
        _pauli = pauli->copy();
        this->_name = "ParametricPauliRotation";
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        for (UINT index = 0; index < target_index_list.size(); ++index) {
            UINT commutation_relation = 0;
            if (pauli_id_list[index] == 1)
                commutation_relation = FLAG_X_COMMUTE;
            else if (pauli_id_list[index] == 2)
                commutation_relation = FLAG_Y_COMMUTE;
            else if (pauli_id_list[index] == 3)
                commutation_relation = FLAG_Z_COMMUTE;
            this->_target_qubit_list.push_back(TargetQubitInfo(
                target_index_list[index], commutation_relation));
        }
    };
    virtual ~ClsParametricPauliRotationGate() { delete _pauli; }
    virtual void update_quantum_state(QuantumStateBase* state) override {
        auto target_index_list = _pauli->get_index_list();
        auto pauli_id_list = _pauli->get_pauli_id_list();
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                multi_qubit_Pauli_rotation_gate_partial_list_host(
                    target_index_list.data(), pauli_id_list.data(),
                    (UINT)target_index_list.size(), _angle, state->data(),
                    state->dim, state->get_cuda_stream(), state->device_number);
            } else {
                multi_qubit_Pauli_rotation_gate_partial_list(
                    target_index_list.data(), pauli_id_list.data(),
                    (UINT)target_index_list.size(), _angle, state->data_c(),
                    state->dim);
            }
#else
            multi_qubit_Pauli_rotation_gate_partial_list(
                target_index_list.data(), pauli_id_list.data(),
                (UINT)target_index_list.size(), _angle, state->data_c(),
                state->dim);
#endif
        } else {
            dm_multi_qubit_Pauli_rotation_gate_partial_list(
                target_index_list.data(), pauli_id_list.data(),
                (UINT)target_index_list.size(), _angle, state->data_c(),
                state->dim);
        }
    };
    virtual ClsParametricPauliRotationGate* copy() const override {
        return new ClsParametricPauliRotationGate(_angle, _pauli);
    };
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        get_Pauli_matrix(matrix, _pauli->get_pauli_id_list());
        std::complex<double> imag_unit(0, 1);
        matrix = cos(_angle / 2) *
                     ComplexMatrix::Identity(matrix.rows(), matrix.cols()) +
                 imag_unit * sin(_angle / 2) * matrix;
    }

    virtual PauliOperator* get_pauli() const { return _pauli; };
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "ParametricPauliRotationGate");
        pt.put("angle", _angle);
        pt.put_child("pauli", _pauli->to_ptree());
        return pt;
    }

    virtual QuantumGateBase* create_gate_whose_qubit_indices_are_replaced(
        const std::vector<UINT>& target_index_list,
        const std::vector<UINT>& control_index_list) const override {
        if (_target_qubit_list.size() != target_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of target_index_list does not match.");
        }
        if (_control_qubit_list.size() != control_index_list.size()) {
            throw InvalidQubitCountException(
                "Error: "
                "QuantumGateBase::create_gate_whose_qubit_indices_is_"
                "replaced\n qubit count of control_index_list does not match.");
        }
        PauliOperator* pauli = new PauliOperator(
            target_index_list, _pauli->get_pauli_id_list(), _pauli->get_coef());
        auto ret = new ClsParametricPauliRotationGate(_angle, pauli);
        delete pauli;
        return ret;
    }

    virtual ClsParametricPauliRotationGate* get_inverse() const override {
        return new ClsParametricPauliRotationGate(-_angle, _pauli);
    };
};
