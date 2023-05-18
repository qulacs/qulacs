#include "parametric_gate.hpp"

QuantumGate_SingleParameter::QuantumGate_SingleParameter(double angle)
    : _angle(angle) {
    _gate_property |= FLAG_PARAMETRIC;
    _parameter_type = 0;
}

void QuantumGate_SingleParameter::set_parameter_value(double value) {
    _angle = value;
}

double QuantumGate_SingleParameter::get_parameter_value() const {
    return _angle;
}

QuantumGate_SingleParameterOneQubitRotation::
    QuantumGate_SingleParameterOneQubitRotation(double angle)
    : QuantumGate_SingleParameter(angle) {}

void QuantumGate_SingleParameterOneQubitRotation::update_quantum_state(
    QuantumStateBase* state) {
    if (state->is_state_vector()) {
#ifdef _USE_GPU
        if (state->get_device_name() == "gpu") {
            if (_update_func_gpu == NULL) {
                throw UndefinedUpdateFuncException(
                    "Error: "
                    "QuantumGate_SingleParameterOneQubitRotation::update_"
                    "quantum_state(QuantumStateBase) : update function is "
                    "undefined");
            }
            _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
                state->data(), state->dim, state->get_cuda_stream(),
                state->device_number);
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
        _update_func(this->_target_qubit_list[0].index(), _angle,
            state->data_c(), state->dim);
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

ClsParametricRXGate::ClsParametricRXGate(UINT target_qubit_index, double angle)
    : QuantumGate_SingleParameterOneQubitRotation(angle) {
    this->_name = "ParametricRX";
    this->_update_func = RX_gate;
    this->_update_func_dm = dm_RX_gate;
#ifdef _USE_GPU
    this->_update_func_gpu = RX_gate_host;
#endif
    this->_target_qubit_list.push_back(
        TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
}

void ClsParametricRXGate::set_matrix(ComplexMatrix& matrix) const {
    matrix = ComplexMatrix::Zero(2, 2);
    matrix << cos(_angle / 2), sin(_angle / 2) * 1.i, sin(_angle / 2) * 1.i,
        cos(_angle / 2);
}

ClsParametricRXGate* ClsParametricRXGate::copy() const {
    return new ClsParametricRXGate(*this);
}

boost::property_tree::ptree ClsParametricRXGate::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "ParametricRXGate");
    pt.put("target_qubit", _target_qubit_list[0].index());
    pt.put("angle", _angle);
    return pt;
}

ClsParametricRXGate* ClsParametricRXGate::get_inverse() const {
    return new ClsParametricRXGate(this->target_qubit_list[0].index(), -_angle);
}

ClsParametricRYGate::ClsParametricRYGate(UINT target_qubit_index, double angle)
    : QuantumGate_SingleParameterOneQubitRotation(angle) {
    this->_name = "ParametricRY";
    this->_update_func = RY_gate;
    this->_update_func_dm = dm_RY_gate;
#ifdef _USE_GPU
    this->_update_func_gpu = RY_gate_host;
#endif
    this->_target_qubit_list.push_back(
        TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
}

void ClsParametricRYGate::set_matrix(ComplexMatrix& matrix) const {
    matrix = ComplexMatrix::Zero(2, 2);
    matrix << cos(_angle / 2), sin(_angle / 2), -sin(_angle / 2),
        cos(_angle / 2);
}

ClsParametricRYGate* ClsParametricRYGate::copy() const {
    return new ClsParametricRYGate(*this);
}

boost::property_tree::ptree ClsParametricRYGate::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "ParametricRYGate");
    pt.put("target_qubit", _target_qubit_list[0].index());
    pt.put("angle", _angle);
    return pt;
}

ClsParametricRYGate* ClsParametricRYGate::get_inverse() const {
    return new ClsParametricRYGate(this->target_qubit_list[0].index(), -_angle);
}

ClsParametricRZGate::ClsParametricRZGate(UINT target_qubit_index, double angle)
    : QuantumGate_SingleParameterOneQubitRotation(angle) {
    this->_name = "ParametricRZ";
    this->_update_func = RZ_gate;
    this->_update_func_dm = dm_RZ_gate;
#ifdef _USE_GPU
    this->_update_func_gpu = RZ_gate_host;
#endif
    this->_target_qubit_list.push_back(
        TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
}

void ClsParametricRZGate::set_matrix(ComplexMatrix& matrix) const {
    matrix = ComplexMatrix::Zero(2, 2);
    matrix << cos(_angle / 2) + 1.i * sin(_angle / 2), 0, 0,
        cos(_angle / 2) - 1.i * sin(_angle / 2);
}

ClsParametricRZGate* ClsParametricRZGate::copy() const {
    return new ClsParametricRZGate(*this);
}

boost::property_tree::ptree ClsParametricRZGate::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "ParametricRZGate");
    pt.put("target_qubit", _target_qubit_list[0].index());
    pt.put("angle", _angle);
    return pt;
}

ClsParametricRZGate* ClsParametricRZGate::get_inverse() const {
    return new ClsParametricRZGate(this->target_qubit_list[0].index(), -_angle);
}

ClsParametricPauliRotationGate::ClsParametricPauliRotationGate(
    double angle, PauliOperator* pauli)
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
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_index_list[index], commutation_relation));
    }
};

ClsParametricPauliRotationGate::~ClsParametricPauliRotationGate() {
    delete _pauli;
}

void ClsParametricPauliRotationGate::update_quantum_state(
    QuantumStateBase* state) {
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
        multi_qubit_Pauli_rotation_gate_partial_list(target_index_list.data(),
            pauli_id_list.data(), (UINT)target_index_list.size(), _angle,
            state->data_c(), state->dim);
#endif
    } else {
        dm_multi_qubit_Pauli_rotation_gate_partial_list(
            target_index_list.data(), pauli_id_list.data(),
            (UINT)target_index_list.size(), _angle, state->data_c(),
            state->dim);
    }
}

ClsParametricPauliRotationGate* ClsParametricPauliRotationGate::copy() const {
    return new ClsParametricPauliRotationGate(_angle, _pauli);
}

void ClsParametricPauliRotationGate::set_matrix(ComplexMatrix& matrix) const {
    get_Pauli_matrix(matrix, _pauli->get_pauli_id_list());
    std::complex<double> imag_unit(0, 1);
    matrix = cos(_angle / 2) *
                 ComplexMatrix::Identity(matrix.rows(), matrix.cols()) +
             imag_unit * sin(_angle / 2) * matrix;
}

PauliOperator* ClsParametricPauliRotationGate::get_pauli() const {
    return _pauli;
}

boost::property_tree::ptree ClsParametricPauliRotationGate::to_ptree() const {
    boost::property_tree::ptree pt;
    pt.put("name", "ParametricPauliRotationGate");
    pt.put("angle", _angle);
    pt.put_child("pauli", _pauli->to_ptree());
    return pt;
}

ClsParametricPauliRotationGate* ClsParametricPauliRotationGate::get_inverse()
    const {
    return new ClsParametricPauliRotationGate(-_angle, _pauli);
}
