#pragma once

#include <cmath>
#include <csim/update_ops.hpp>
#include <csim/update_ops_dm.hpp>

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif

/**
 * \~japanese-en 1量子ビットを対象とする回転角固定のゲートのクラス
 */

static void Igate_idling(UINT, CTYPE*, ITYPE){};
static void Igate_idling_gpu(UINT, void*, ITYPE, void*, UINT){};
static void Igate_idling_mpi(UINT, CTYPE*, ITYPE, UINT){};

class ClsOneQubitGate : public QuantumGateBase {
protected:
    using UpdateFunc = void (*)(UINT, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func;
    UpdateFunc _update_func_dm;
    UpdateFuncGpu _update_func_gpu;
    UpdateFuncMpi _update_func_mpi;
    ComplexMatrix _matrix_element;

public:
    explicit ClsOneQubitGate(){};
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                _update_func_gpu(this->target_qubit_list[0].index(),
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                _update_func_mpi(this->_target_qubit_list[0].index(),
                    state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                _update_func(this->_target_qubit_list[0].index(),
                    state->data_c(), state->dim);
            }
        } else {
            _update_func_dm(this->_target_qubit_list[0].index(),
                state->data_c(), state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsOneQubitGate* copy() const override {
        return new ClsOneQubitGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }

    void IGateinit(UINT target_qubit_index) {
        this->_update_func = Igate_idling;
        this->_update_func_dm = Igate_idling;
        this->_update_func_gpu = Igate_idling_gpu;
        this->_update_func_mpi = Igate_idling_mpi;
        this->_name = "I";
        this->_target_qubit_list.push_back(TargetQubitInfo(target_qubit_index,
            FLAG_X_COMMUTE | FLAG_Y_COMMUTE | FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, 1;
    }

    void XGateinit(UINT target_qubit_index) {
        this->_update_func = X_gate;
        this->_update_func_dm = dm_X_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = X_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = X_gate_mpi;
#endif
        this->_name = "X";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, 1, 1, 0;
    }

    void YGateinit(UINT target_qubit_index) {
        this->_update_func = Y_gate;
        this->_update_func_dm = dm_Y_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = Y_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = Y_gate_mpi;
#endif
        this->_name = "Y";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, -1.i, 1.i, 0;
    }

    void ZGateinit(UINT target_qubit_index) {
        this->_update_func = Z_gate;
        this->_update_func_dm = dm_Z_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = Z_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = Z_gate_mpi;
#endif
        this->_name = "Z";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_PAULI | FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, -1;
    }

    void HGateinit(UINT target_qubit_index) {
        this->_update_func = H_gate;
        this->_update_func_dm = dm_H_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = H_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = H_gate_mpi;
#endif
        this->_name = "H";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 1, 1, -1;
        this->_matrix_element /= sqrt(2.);
    }

    void SGateinit(UINT target_qubit_index) {
        this->_update_func = S_gate;
        this->_update_func_dm = dm_S_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = S_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = S_gate_mpi;
#endif
        this->_name = "S";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, 1.i;
    }

    void SdagGateinit(UINT target_qubit_index) {
        this->_update_func = Sdag_gate;
        this->_update_func_dm = dm_Sdag_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = Sdag_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = Sdag_gate_mpi;
#endif
        this->_name = "Sdag";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, -1.i;
    }

    void TGateinit(UINT target_qubit_index) {
        this->_update_func = T_gate;
        this->_update_func_dm = dm_T_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = T_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = T_gate_mpi;
#endif
        this->_name = "T";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, (1. + 1.i) / sqrt(2.);
    }

    void TdagGateinit(UINT target_qubit_index) {
        this->_update_func = Tdag_gate;
        this->_update_func_dm = dm_Tdag_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = Tdag_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = Tdag_gate_mpi;
#endif
        this->_name = "Tdag";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_gate_property = FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, (1. - 1.i) / sqrt(2.);
    }

    void sqrtXGateinit(UINT target_qubit_index) {
        this->_update_func = sqrtX_gate;
        this->_update_func_dm = dm_sqrtX_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = sqrtX_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = sqrtX_gate_mpi;
#endif
        this->_name = "sqrtX";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5 + 0.5i, 0.5 - 0.5i, 0.5 - 0.5i, 0.5 + 0.5i;
    }
    void sqrtXdagGateinit(UINT target_qubit_index) {
        this->_update_func = sqrtXdag_gate;
        this->_update_func_dm = dm_sqrtXdag_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = sqrtXdag_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = sqrtXdag_gate_mpi;
#endif
        this->_name = "sqrtXdag";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5 - 0.5i, 0.5 + 0.5i, 0.5 + 0.5i, 0.5 - 0.5i;
    }
    void sqrtYGateinit(UINT target_qubit_index) {
        this->_update_func = sqrtY_gate;
        this->_update_func_dm = dm_sqrtY_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = sqrtY_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = sqrtY_gate_mpi;
#endif
        this->_name = "sqrtY";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5 + 0.5i, -0.5 - 0.5i, 0.5 + 0.5i,
            0.5 + 0.5i;
    }
    void sqrtYdagGateinit(UINT target_qubit_index) {
        this->_update_func = sqrtYdag_gate;
        this->_update_func_dm = dm_sqrtYdag_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = sqrtYdag_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = sqrtYdag_gate_mpi;
#endif
        this->_name = "sqrtYdag";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0.5 - 0.5i, 0.5 - 0.5i, -0.5 + 0.5i,
            0.5 - 0.5i;
    }
    void P0Gateinit(UINT target_qubit_index) {
        this->_update_func = P0_gate;
        this->_update_func_dm = dm_P0_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = P0_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = P0_gate_mpi;
#endif
        this->_name = "Projection-0";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, 0;
    }
    void P1Gateinit(UINT target_qubit_index) {
        this->_update_func = P1_gate;
        this->_update_func_dm = dm_P1_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = P1_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = P1_gate_mpi;
#endif
        this->_name = "Projection-1";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, 0));
        this->_gate_property = FLAG_CLIFFORD | FLAG_GAUSSIAN;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, 0, 0, 1;
    }

    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", _name + "Gate");
        pt.add("target_qubit", _target_qubit_list[0].index());
        return pt;
    }
    virtual ClsOneQubitGate* get_inverse(void) const override;
};

/**
 * \~japanese-en 1量子ビットを対象とする回転ゲートのクラス
 */
class ClsOneQubitRotationGate : public QuantumGateBase {
protected:
    using UpdateFunc = void (*)(UINT, double, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, double, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, double, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func;
    UpdateFunc _update_func_dm;
    UpdateFuncGpu _update_func_gpu;
    UpdateFuncMpi _update_func_mpi;
    ComplexMatrix _matrix_element;
    double _angle;

public:
    explicit ClsOneQubitRotationGate(){};
    explicit ClsOneQubitRotationGate(double angle) : _angle(angle){};
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                _update_func_gpu(this->_target_qubit_list[0].index(), _angle,
                    state->data(), state->dim, state->get_cuda_stream(),
                    state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                _update_func_mpi(this->_target_qubit_list[0].index(), _angle,
                    state->data_c(), state->dim, state->inner_qc);
            } else
#endif
            {
                _update_func(this->_target_qubit_list[0].index(), _angle,
                    state->data_c(), state->dim);
            }
        } else {
            _update_func_dm(this->_target_qubit_list[0].index(), _angle,
                state->data_c(), state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsOneQubitRotationGate* copy() const override {
        return new ClsOneQubitRotationGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }

    void RXGateinit(UINT target_qubit_index, double angle) {
        this->_angle = angle;
        this->_update_func = RX_gate;
        this->_update_func_dm = dm_RX_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RX_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RX_gate_mpi;
#endif
        this->_name = "X-rotation";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle / 2), sin(_angle / 2) * 1.i,
            sin(_angle / 2) * 1.i, cos(_angle / 2);
    }

    void RYGateinit(UINT target_qubit_index, double angle) {
        this->_angle = angle;
        this->_update_func = RY_gate;
        this->_update_func_dm = dm_RY_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RY_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RY_gate_mpi;
#endif
        this->_name = "Y-rotation";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Y_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle / 2), sin(_angle / 2),
            -sin(_angle / 2), cos(_angle / 2);
    }

    void RZGateinit(UINT target_qubit_index, double angle) {
        this->_angle = angle;
        this->_update_func = RZ_gate;
        this->_update_func_dm = dm_RZ_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = RZ_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = RZ_gate_mpi;
#endif
        this->_name = "Z-rotation";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << cos(_angle / 2) + 1.i * sin(_angle / 2), 0, 0,
            cos(_angle / 2) - 1.i * sin(_angle / 2);
    }

    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", _name + "Gate");
        pt.add("target_qubit", _target_qubit_list[0].index());
        pt.add("angle", _angle);
        return pt;
    }
    virtual ClsOneQubitRotationGate* get_inverse(void) const override;
};

using QuantumGate_OneQubit = ClsOneQubitGate;
using QuantumGate_OneQubitRotation = ClsOneQubitRotationGate;
