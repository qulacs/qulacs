#include "gate.hpp"

void QuantumGateBasic::_update_state_vector_cpu_special(QuantumStateBase* state) const {
    if (_special_func_type == GateI) {
        // pass
    }
    else if (_special_func_type == GateX) {
        X_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateY) {
        Y_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateZ) {
        Z_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSqrtX) {
        sqrtX_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSqrtXdag) {
        sqrtXdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSqrtY) {
        sqrtY_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSqrtYdag) {
        sqrtYdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateH) {
        H_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateS) {
        S_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSdag) {
        Sdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateT) {
        T_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateTdag) {
        Tdag_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateP0) {
        P0_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateP1) {
        P1_gate(_target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateRX) {
        // invert
        RX_gate(_target_qubit_index[0], - _rotation_angle, state->data_c(), state->dim);
    }
    else if (_special_func_type == GateRY) {
        // invert
        RY_gate(_target_qubit_index[0], - _rotation_angle, state->data_c(), state->dim);
    }
    else if (_special_func_type == GateRZ) {
        // invert
        RZ_gate(_target_qubit_index[0], - _rotation_angle, state->data_c(), state->dim);
    }
    else if (_special_func_type == GateCX) {
        CNOT_gate(_control_qubit_index[0], _target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateCZ) {
        CZ_gate(_control_qubit_index[0], _target_qubit_index[0], state->data_c(), state->dim);
    }
    else if (_special_func_type == GateSWAP) {
        SWAP_gate(_target_qubit_index[0], _target_qubit_index[1], state->data_c(), state->dim);
    }
    else {
        throw std::invalid_argument("Unsupported special gate");
    }
}


namespace gate {
    DllExport QuantumGateBasic* Identity(UINT target_qubit) {
        ComplexMatrix mat = ComplexMatrix::Identity(2,2);
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_X | FLAG_COMMUTE_Y | FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateI);
        return ptr;
    }
    DllExport QuantumGateBasic* X(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0, 1, 1, 0;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_X});
        ptr->set_special_func_type(GateX);
        return ptr;
    }
    DllExport QuantumGateBasic* Y(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0, -1.i, 1.i, 0;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_Y});
        ptr->set_special_func_type(GateY);
        return ptr;
    }
    DllExport QuantumGateBasic* Z(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1, 0, 0, -1;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateZ);
        return ptr;
    }
    DllExport QuantumGateBasic* sqrtX(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_X});
        ptr->set_special_func_type(GateSqrtX);
        return ptr;
    }
    DllExport QuantumGateBasic* sqrtXdag(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat.adjoint(), {FLAG_COMMUTE_X});
        ptr->set_special_func_type(GateSqrtXdag);
        return ptr;
    }
    DllExport QuantumGateBasic* sqrtY(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_Y});
        ptr->set_special_func_type(GateSqrtY);
        return ptr;
    }
    DllExport QuantumGateBasic* sqrtYdag(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat.adjoint(), {FLAG_COMMUTE_Y});
        ptr->set_special_func_type(GateSqrtYdag);
        return ptr;
    }
    DllExport QuantumGateBasic* RX(UINT target_qubit, double rotation_angle) {
        auto ptr = QuantumGateBasic::PauliMatrixGate({ target_qubit }, { PAULI_ID_X }, rotation_angle);
        ptr->set_special_func_type(GateRX);
        return ptr;
    }
    DllExport QuantumGateBasic* RY(UINT target_qubit, double rotation_angle) {
        auto ptr = QuantumGateBasic::PauliMatrixGate({ target_qubit }, { PAULI_ID_Y }, rotation_angle);
        ptr->set_special_func_type(GateRY);
        return ptr;
    }
    DllExport QuantumGateBasic* RZ(UINT target_qubit, double rotation_angle) {
        auto ptr = QuantumGateBasic::PauliMatrixGate({ target_qubit }, { PAULI_ID_Z }, rotation_angle);
        ptr->set_special_func_type(GateRZ);
        return ptr;
    }
    DllExport QuantumGateBasic* H(UINT target_qubit) {
        double invsqrt2 = 1. / sqrt(2.);
        ComplexMatrix mat(2, 2);
        mat << invsqrt2, invsqrt2, invsqrt2, -invsqrt2;
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {});
        ptr->set_special_func_type(GateH);
        return ptr;
    }
    DllExport QuantumGateBasic* S(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1., 0., 0., 1.i;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateS);
        return ptr;
    }
    DllExport QuantumGateBasic* HS(UINT target_qubit) {
        double invsqrt2 = 1. / sqrt(2.);
        ComplexMatrix mat(2, 2);
        mat << invsqrt2, invsqrt2*1.i, invsqrt2, -invsqrt2*1.i;
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {});
        return ptr;
    }
    DllExport QuantumGateBasic* Sdag(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1., 0., 0., -1.i;
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateSdag);
        return ptr;
    }
    DllExport QuantumGateBasic* T(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1., 0., 0., (1.+1.i)/sqrt(2.);
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateT);
        return ptr;
    }
    DllExport QuantumGateBasic* Tdag(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1., 0., 0., (1. - 1.i) / sqrt(2.);
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateTdag);
        return ptr;
    }
    DllExport QuantumGateBasic* P0(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 1., 0., 0., 0;
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateP0);
        return ptr;
    }
    DllExport QuantumGateBasic* P1(UINT target_qubit) {
        ComplexMatrix mat(2, 2);
        mat << 0., 0., 0., 1;
        auto ptr = QuantumGateBasic::DenseMatrixGate(
            { target_qubit }, mat, {FLAG_COMMUTE_Z});
        ptr->set_special_func_type(GateP1);
        return ptr;
    }
    DllExport QuantumGateBasic* CX(UINT control_qubit, UINT target_qubit) {
        ComplexMatrix mat;
        get_Pauli_matrix(mat, { PAULI_ID_X });
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, { FLAG_COMMUTE_X });
        ptr->add_control_qubit(control_qubit, 1);
        ptr->set_special_func_type(GateCX);
        return ptr;
    }
    DllExport QuantumGateBasic* CNOT(UINT control_qubit, UINT target_qubit) {
        return CX(control_qubit, target_qubit);
    }
    DllExport QuantumGateBasic* CY(UINT control_qubit, UINT target_qubit) {
        ComplexMatrix mat;
        get_Pauli_matrix(mat, { PAULI_ID_Y });
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, { FLAG_COMMUTE_Y });
        ptr->add_control_qubit(control_qubit, 1);
        ptr->set_special_func_type(GateCY);
        return ptr;
    }
    DllExport QuantumGateBasic* CZ(UINT control_qubit, UINT target_qubit) {
        ComplexMatrix mat;
        get_Pauli_matrix(mat, { PAULI_ID_Z });
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, { FLAG_COMMUTE_Z });
        ptr->add_control_qubit(control_qubit, 1);
        ptr->set_special_func_type(GateCZ);
        return ptr;
    }
    DllExport QuantumGateBasic* SWAP(UINT target_qubit1, UINT target_qubit2) {
        ComplexMatrix mat(4,4);
        mat <<
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1;                
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit1, target_qubit2 }, mat, { });
        ptr->set_special_func_type(GateSWAP);
        return ptr;
    }
    DllExport QuantumGateBasic* ISWAP(UINT target_qubit1, UINT target_qubit2) {
        ComplexMatrix mat(4, 4);
        mat <<
            1, 0, 0, 0,
            0, 0, 1.i, 0,
            0, 1.i, 0, 0,
            0, 0, 0, 1;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit1, target_qubit2 }, mat, { });
        //ptr->set_special_func_type(GateISWAP);
        return ptr;
    }
    DllExport QuantumGateBasic* Toffoli(UINT control_qubit1, UINT control_qubit2, UINT target_qubit) {
        ComplexMatrix mat;
        get_Pauli_matrix(mat, { PAULI_ID_X });
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit }, mat, { FLAG_COMMUTE_X });
        ptr->add_control_qubit(control_qubit1, 1);
        ptr->add_control_qubit(control_qubit2, 1);
        return ptr;
    }
    DllExport QuantumGateBasic* Fredkin(UINT control_qubit, UINT target_qubit1, UINT target_qubit2) {
        ComplexMatrix mat(4, 4);
        mat <<
            1, 0, 0, 0,
            0, 0, 1, 0,
            0, 1, 0, 0,
            0, 0, 0, 1;
        auto ptr = QuantumGateBasic::DenseMatrixGate({ target_qubit1, target_qubit2 }, mat, { });
        ptr->add_control_qubit(control_qubit, 1);
        return ptr;
    }




    DllExport QuantumGateWrapped* DepolarizingNoise(UINT index, double prob) {
        auto ptr = QuantumGateWrapped::ProbabilisticGate(
            { gate::Identity(index), gate::X(index), gate::Y(index), gate::Z(index) },
            { 1 - prob, prob / 3, prob / 3, prob / 3 }, true
        );
        return ptr;
    }
    DllExport QuantumGateWrapped* IndependentXZNoise(UINT index, double prob) {
        auto ptr = QuantumGateWrapped::ProbabilisticGate(
            { gate::Identity(index), gate::X(index), gate::Z(index), gate::Y(index) },
            { (1 - prob)*(1 - prob), prob*(1-prob), (1-prob)*prob, prob*prob}, true
        );
        return ptr;
    }

    DllExport QuantumGateWrapped* TwoQubitDepolarizingNoise(UINT index1, UINT index2, double prob) {
        std::vector<QuantumGateBase*> gates;
        std::vector<double> probs;
        probs.push_back(1 - prob);
        gates.push_back(gate::Identity(index1));
        for (UINT i = 1; i < 16; ++i) {
            auto gate = QuantumGateBasic::PauliMatrixGate({ index1, index2 }, { i % 4, i / 4 }, 0.);
            gates.push_back(gate);
            probs.push_back(prob / 15);
        }
        auto ptr = QuantumGateWrapped::ProbabilisticGate(gates, probs, true);
        return ptr;
    }
    DllExport QuantumGateWrapped* BitFlipNoise(UINT index, double prob) {
        auto ptr = QuantumGateWrapped::ProbabilisticGate(
            { gate::Identity(index), gate::X(index) },
            { 1 - prob, prob }, true
        );
        return ptr;
    }
    DllExport QuantumGateWrapped* DephasingNoise(UINT index, double prob) {
        auto ptr = QuantumGateWrapped::ProbabilisticGate(
            { gate::Identity(index), gate::Z(index) },
            { 1 - prob, prob }, true
        );
        return ptr;
    }
}
