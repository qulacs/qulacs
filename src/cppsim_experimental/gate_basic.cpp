
#include "gate_basic.hpp"

void QuantumGateBasic::_update_state_vector_cpu_special(
    QuantumStateBase* state) const {
    if (_special_func_type == GateI) {
        // pass
    } else if (_special_func_type == GateX) {
        X_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateY) {
        Y_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateZ) {
        Z_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateSqrtX) {
        sqrtX_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateSqrtXdag) {
        sqrtXdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateSqrtY) {
        sqrtY_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateSqrtYdag) {
        sqrtYdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateH) {
        H_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateS) {
        S_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateSdag) {
        Sdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateT) {
        T_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateTdag) {
        Tdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateP0) {
        P0_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateP1) {
        P1_gate(_target_qubit_index[0], state->data_c(), state->dim);
    } else if (_special_func_type == GateRX) {
        // invert
        RX_gate(_target_qubit_index[0], -_rotation_angle, state->data_c(),
            state->dim);
    } else if (_special_func_type == GateRY) {
        // invert
        RY_gate(_target_qubit_index[0], -_rotation_angle, state->data_c(),
            state->dim);
    } else if (_special_func_type == GateRZ) {
        // invert
        RZ_gate(_target_qubit_index[0], -_rotation_angle, state->data_c(),
            state->dim);
    } else if (_special_func_type == GateCX) {
        CNOT_gate(_control_qubit_index[0], _target_qubit_index[0],
            state->data_c(), state->dim);
    } else if (_special_func_type == GateCZ) {
        CZ_gate(_control_qubit_index[0], _target_qubit_index[0],
            state->data_c(), state->dim);
    } else if (_special_func_type == GateSWAP) {
        SWAP_gate(_target_qubit_index[0], _target_qubit_index[1],
            state->data_c(), state->dim);
    } else {
        throw std::invalid_argument("Unsupported special gate");
    }
}

void QuantumGateBasic::_update_state_vector_cpu_general(QuantumStateBase* state) const {
    if (_matrix_type == DenseMatrix) {
        const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
            this->_dense_matrix_element.data());
        // single qubit dense matrix gate
        if (_target_qubit_index.size() == 1) {
            // no control qubit
            if (_control_qubit_index.size() == 0) {
                single_qubit_dense_matrix_gate(_target_qubit_index[0],
                    matrix_ptr, state->data_c(), state->dim);
            }
            // single control qubit
            else if (_control_qubit_index.size() == 1) {
                single_qubit_control_single_qubit_dense_matrix_gate(
                    _control_qubit_index[0], _control_qubit_value[0],
                    _target_qubit_index[0], matrix_ptr, state->data_c(),
                    state->dim);
            }
            // multiple control qubits
            else {
                multi_qubit_control_single_qubit_dense_matrix_gate(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)(_control_qubit_index.size()),
                    _target_qubit_index[0], matrix_ptr, state->data_c(),
                    state->dim);
            }
        }

        // multi qubit dense matrix gate
        else {
            // no control qubit
            if (_control_qubit_index.size() == 0) {
                multi_qubit_dense_matrix_gate(_target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()), matrix_ptr,
                    state->data_c(), state->dim);
            }
            // single control qubit
            else if (_control_qubit_index.size() == 1) {
                single_qubit_control_multi_qubit_dense_matrix_gate(
                    _control_qubit_index[0], _control_qubit_value[0],
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()), matrix_ptr,
                    state->data_c(), state->dim);
            }
            // multiple control qubit
            else {
                multi_qubit_control_multi_qubit_dense_matrix_gate(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)(_control_qubit_index.size()),
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()), matrix_ptr,
                    state->data_c(), state->dim);
            }
        }
    }
    else if (_matrix_type == DiagonalMatrix) {
        const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
            this->_diagonal_matrix_element.data());
        if (_target_qubit_index.size() == 1)
            single_qubit_diagonal_matrix_gate(_target_qubit_index[0],
                matrix_ptr, state->data_c(), state->dim);
        else
            multi_qubit_diagonal_matrix_gate(_target_qubit_index.data(),
            (UINT)_target_qubit_index.size(), matrix_ptr,
                state->data_c(), state->dim);
    }
    else if (_matrix_type == SparseMatrix) {
        multi_qubit_sparse_matrix_gate_eigen(_target_qubit_index.data(),
            (UINT)(_target_qubit_index.size()),
            this->_sparse_matrix_element, state->data_c(), state->dim);
    }
    else if (_matrix_type == PauliMatrix) {
        if (_target_qubit_index.size() == 1) {
            if (fabs(_rotation_angle) < 1e-16) {
                single_qubit_Pauli_gate(_target_qubit_index[0],
                    _pauli_id[0], state->data_c(), state->dim);
            }
            else {
                // invert
                single_qubit_Pauli_rotation_gate(_target_qubit_index[0],
                    _pauli_id[0], -_rotation_angle, state->data_c(),
                    state->dim);
            }
        }
        else {
            if (fabs(_rotation_angle) < 1e-16) {
                multi_qubit_Pauli_gate_partial_list(
                    _target_qubit_index.data(), _pauli_id.data(),
                    (UINT)_target_qubit_index.size(), state->data_c(),
                    state->dim);
            }
            else {
                // invert
                multi_qubit_Pauli_rotation_gate_partial_list(
                    _target_qubit_index.data(), _pauli_id.data(),
                    (UINT)_target_qubit_index.size(), -_rotation_angle,
                    state->data_c(), state->dim);
            }
        }
    }
}

void QuantumGateBasic::_update_density_matrix_cpu_general(QuantumStateBase* state) const {
    if (_matrix_type == DenseMatrix) {
        const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
            this->_dense_matrix_element.data());
        if (_control_qubit_index.size() == 0) {
            if (_target_qubit_index.size() == 1) {
                dm_single_qubit_dense_matrix_gate(_target_qubit_index[0],
                    matrix_ptr, state->data_c(), state->dim);
            }
            else {
                dm_multi_qubit_dense_matrix_gate(_target_qubit_index.data(),
                    (UINT)_target_qubit_index.size(), matrix_ptr,
                    state->data_c(), state->dim);
            }
        }
        else {
            if (_target_qubit_index.size() == 1) {
                dm_multi_qubit_control_single_qubit_dense_matrix_gate(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)_control_qubit_index.size(),
                    _target_qubit_index[0], matrix_ptr, state->data_c(),
                    state->dim);
            }
            else {
                dm_multi_qubit_control_multi_qubit_dense_matrix_gate(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)_control_qubit_index.size(),
                    _target_qubit_index.data(),
                    (UINT)_target_qubit_index.size(), matrix_ptr,
                    state->data_c(), state->dim);
            }
        }
    }
    else {
        throw std::invalid_argument(
            "Only DenseMatrix gate type is supported for density matrix");
    }
}

#ifdef _USE_GPU
void QuantumGateBasic::_update_state_vector_gpu(QuantumStateBase* state) {
    if (_matrix_type == DenseMatrix) {
        const CTYPE* matrix_ptr = reinterpret_cast<const CTYPE*>(
            this->_dense_matrix_element.data());
        // single qubit dense matrix gate
        if (_target_qubit_index.size() == 1) {
            // no control qubit
            if (_control_qubit_index.size() == 0) {
                single_qubit_dense_matrix_gate_host(_target_qubit_index[0],
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            }
            // single control qubit
            else if (_control_qubit_index.size() == 1) {
                single_qubit_control_single_qubit_dense_matrix_gate_host(
                    _control_qubit_index[0], _control_qubit_value[0],
                    _target_qubit_index[0], (const CPPCTYPE*)matrix_ptr,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            }
            // multiple control qubits
            else {
                multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)(_control_qubit_index.size()),
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()),
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            }
        }

        // multi qubit dense matrix gate
        else {
            // no control qubit
            if (_control_qubit_index.size() == 0) {
                multi_qubit_dense_matrix_gate_host(
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()),
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            }
            // single control qubit
            else if (_control_qubit_index.size() == 1) {
                single_qubit_control_multi_qubit_dense_matrix_gate_host(
                    _control_qubit_index[0], _control_qubit_value[0],
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()),
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            }
            // multiple control qubit
            else {
                multi_qubit_control_multi_qubit_dense_matrix_gate_host(
                    _control_qubit_index.data(),
                    _control_qubit_value.data(),
                    (UINT)(_control_qubit_index.size()),
                    _target_qubit_index.data(),
                    (UINT)(_target_qubit_index.size()),
                    (const CPPCTYPE*)matrix_ptr, state->data(), state->dim,
                    state->get_cuda_stream(), state->device_number);
            }
        }
    }
    else {
        throw std::invalid_argument(
            "Only DenseMatrix gate type is supported for density matrix");
    }
}
void _update_density_matrix_gpu(QuantumStateBase* state) {
    throw std::runtime_error(
        "Density matrix simulation is not supported on GPU.");
}
#endif

namespace gate {
DllExport QuantumGateBasic* Identity(UINT target_qubit) {
    ComplexMatrix mat = ComplexMatrix::Identity(2, 2);
    auto ptr = QuantumGateBasic::DenseMatrixGate({target_qubit}, mat,
        {FLAG_COMMUTE_X | FLAG_COMMUTE_Y | FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateI);
    return ptr;
}
DllExport QuantumGateBasic* X(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0, 1, 1, 0;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_X});
    ptr->_set_special_func_type(GateX);
    return ptr;
}
DllExport QuantumGateBasic* Y(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0, -1.i, 1.i, 0;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Y});
    ptr->_set_special_func_type(GateY);
    return ptr;
}
DllExport QuantumGateBasic* Z(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1, 0, 0, -1;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateZ);
    return ptr;
}
DllExport QuantumGateBasic* sqrtX(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_X});
    ptr->_set_special_func_type(GateSqrtX);
    return ptr;
}
DllExport QuantumGateBasic* sqrtXdag(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat.adjoint(), {FLAG_COMMUTE_X});
    ptr->_set_special_func_type(GateSqrtXdag);
    return ptr;
}
DllExport QuantumGateBasic* sqrtY(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Y});
    ptr->_set_special_func_type(GateSqrtY);
    return ptr;
}
DllExport QuantumGateBasic* sqrtYdag(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat.adjoint(), {FLAG_COMMUTE_Y});
    ptr->_set_special_func_type(GateSqrtYdag);
    return ptr;
}
DllExport QuantumGateBasic* RX(UINT target_qubit, double rotation_angle) {
    auto ptr = QuantumGateBasic::PauliMatrixGate(
        {target_qubit}, {PAULI_ID_X}, rotation_angle);
    ptr->_set_special_func_type(GateRX);
    return ptr;
}
DllExport QuantumGateBasic* RY(UINT target_qubit, double rotation_angle) {
    auto ptr = QuantumGateBasic::PauliMatrixGate(
        {target_qubit}, {PAULI_ID_Y}, rotation_angle);
    ptr->_set_special_func_type(GateRY);
    return ptr;
}
DllExport QuantumGateBasic* RZ(UINT target_qubit, double rotation_angle) {
    auto ptr = QuantumGateBasic::PauliMatrixGate(
        {target_qubit}, {PAULI_ID_Z}, rotation_angle);
    ptr->_set_special_func_type(GateRZ);
    return ptr;
}
DllExport QuantumGateBasic* H(UINT target_qubit) {
    double invsqrt2 = 1. / sqrt(2.);
    ComplexMatrix mat(2, 2);
    mat << invsqrt2, invsqrt2, invsqrt2, -invsqrt2;
    auto ptr = QuantumGateBasic::DenseMatrixGate({target_qubit}, mat, {});
    ptr->_set_special_func_type(GateH);
    return ptr;
}
DllExport QuantumGateBasic* S(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1., 0., 0., 1.i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateS);
    return ptr;
}
DllExport QuantumGateBasic* HS(UINT target_qubit) {
    double invsqrt2 = 1. / sqrt(2.);
    ComplexMatrix mat(2, 2);
    mat << invsqrt2, invsqrt2 * 1.i, invsqrt2, -invsqrt2 * 1.i;
    auto ptr = QuantumGateBasic::DenseMatrixGate({target_qubit}, mat, {});
    return ptr;
}
DllExport QuantumGateBasic* Sdag(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1., 0., 0., -1.i;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateSdag);
    return ptr;
}
DllExport QuantumGateBasic* T(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1., 0., 0., (1. + 1.i) / sqrt(2.);
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateT);
    return ptr;
}
DllExport QuantumGateBasic* Tdag(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1., 0., 0., (1. - 1.i) / sqrt(2.);
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateTdag);
    return ptr;
}
DllExport QuantumGateBasic* P0(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 1., 0., 0., 0;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateP0);
    return ptr;
}
DllExport QuantumGateBasic* P1(UINT target_qubit) {
    ComplexMatrix mat(2, 2);
    mat << 0., 0., 0., 1;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->_set_special_func_type(GateP1);
    return ptr;
}
DllExport QuantumGateBasic* CX(UINT control_qubit, UINT target_qubit) {
    ComplexMatrix mat;
    get_Pauli_matrix(mat, {PAULI_ID_X});
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_X});
    ptr->add_control_qubit(control_qubit, 1);
    ptr->_set_special_func_type(GateCX);
    return ptr;
}
DllExport QuantumGateBasic* CNOT(UINT control_qubit, UINT target_qubit) {
    return CX(control_qubit, target_qubit);
}
DllExport QuantumGateBasic* CY(UINT control_qubit, UINT target_qubit) {
    ComplexMatrix mat;
    get_Pauli_matrix(mat, {PAULI_ID_Y});
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Y});
    ptr->add_control_qubit(control_qubit, 1);
    ptr->_set_special_func_type(GateCY);
    return ptr;
}
DllExport QuantumGateBasic* CZ(UINT control_qubit, UINT target_qubit) {
    ComplexMatrix mat;
    get_Pauli_matrix(mat, {PAULI_ID_Z});
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_Z});
    ptr->add_control_qubit(control_qubit, 1);
    ptr->_set_special_func_type(GateCZ);
    return ptr;
}
DllExport QuantumGateBasic* SWAP(UINT target_qubit1, UINT target_qubit2) {
    ComplexMatrix mat(4, 4);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit1, target_qubit2}, mat, {});
    ptr->_set_special_func_type(GateSWAP);
    return ptr;
}
DllExport QuantumGateBasic* ISWAP(UINT target_qubit1, UINT target_qubit2) {
    ComplexMatrix mat(4, 4);
    mat << 1, 0, 0, 0, 0, 0, 1.i, 0, 0, 1.i, 0, 0, 0, 0, 0, 1;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit1, target_qubit2}, mat, {});
    return ptr;
}
DllExport QuantumGateBasic* Toffoli(
    UINT control_qubit1, UINT control_qubit2, UINT target_qubit) {
    ComplexMatrix mat;
    get_Pauli_matrix(mat, {PAULI_ID_X});
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit}, mat, {FLAG_COMMUTE_X});
    ptr->add_control_qubit(control_qubit1, 1);
    ptr->add_control_qubit(control_qubit2, 1);
    return ptr;
}
DllExport QuantumGateBasic* Fredkin(
    UINT control_qubit, UINT target_qubit1, UINT target_qubit2) {
    ComplexMatrix mat(4, 4);
    mat << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    auto ptr = QuantumGateBasic::DenseMatrixGate(
        {target_qubit1, target_qubit2}, mat, {});
    ptr->add_control_qubit(control_qubit, 1);
    return ptr;
}
}  // namespace gate
