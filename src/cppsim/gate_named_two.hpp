#pragma once

#include <csim/update_ops.hpp>

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

/**
 * \~japanese-en 2量子ビットを対象とする回転角固定のゲートのクラス
 */
class ClsTwoQubitGate : public QuantumGateBase {
protected:
    using UpdateFunc = void (*)(UINT, UINT, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, UINT, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, UINT, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func;
    UpdateFunc _update_func_dm;
    UpdateFuncGpu _update_func_gpu;
    UpdateFuncMpi _update_func_mpi;
    ComplexMatrix _matrix_element;

public:
    explicit ClsTwoQubitGate(){};
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                _update_func_gpu(this->_target_qubit_list[0].index(),
                    this->_target_qubit_list[1].index(), state->data(),
                    state->dim, state->get_cuda_stream(), state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                _update_func_mpi(this->_target_qubit_list[0].index(),
                    this->_target_qubit_list[1].index(), state->data_c(),
                    state->dim, state->inner_qc);
            } else
#endif
            {
                _update_func(this->_target_qubit_list[0].index(),
                    this->_target_qubit_list[1].index(), state->data_c(),
                    state->dim);
            }
        } else {
            _update_func_dm(this->_target_qubit_list[0].index(),
                this->_target_qubit_list[1].index(), state->data_c(),
                state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsTwoQubitGate* copy() const override {
        return new ClsTwoQubitGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }

    void SWAPGateinit(UINT target_qubit_index1, UINT target_qubit_index2) {
        this->_update_func = SWAP_gate;
        this->_update_func_dm = dm_SWAP_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = SWAP_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = SWAP_gate_mpi;
#endif
        this->_name = "SWAP";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index1, 0));
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index2, 0));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(4, 4);
        this->_matrix_element << 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1;
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", _name + "Gate");
        std::vector<UINT> target_qubit_list_uint;
        std::transform(_target_qubit_list.begin(), _target_qubit_list.end(),
            std::back_inserter(target_qubit_list_uint),
            [](const TargetQubitInfo& qubit_info) {
                return qubit_info.index();
            });
        pt.add_child(
            "target_qubit_list", ptree::to_ptree(target_qubit_list_uint));
        return pt;
    }

    virtual ClsTwoQubitGate* get_inverse(void) const override {
        if (this->_name == "SWAP") {
            return this->copy();
        }
        throw NotImplementedException(
            "Inverse of " + this->_name + " gate is not Implemented");
    }
    // 現状SWAPゲートしかないので、自身がget_inverseになるが、　そうでないゲートが追加されたときの保険として、　判定をする
};

/**
 * \~japanese-en
 * 1量子ビットを対象とし1量子ビットにコントロールされる回転角固定のゲートのクラス
 */
class ClsOneControlOneTargetGate : public QuantumGateBase {
protected:
    using UpdateFunc = void (*)(UINT, UINT, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, UINT, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, UINT, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func;
    UpdateFunc _update_func_dm;
    UpdateFuncGpu _update_func_gpu;
    UpdateFuncMpi _update_func_mpi;
    ComplexMatrix _matrix_element;

public:
    explicit ClsOneControlOneTargetGate(){};
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                _update_func_gpu(this->_control_qubit_list[0].index(),
                    this->_target_qubit_list[0].index(), state->data(),
                    state->dim, state->get_cuda_stream(), state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                _update_func_mpi(this->_control_qubit_list[0].index(),
                    this->_target_qubit_list[0].index(), state->data_c(),
                    state->dim, state->inner_qc);
            } else
#endif
            {
                _update_func(this->_control_qubit_list[0].index(),
                    this->_target_qubit_list[0].index(), state->data_c(),
                    state->dim);
            }
        } else {
            _update_func_dm(this->_control_qubit_list[0].index(),
                this->_target_qubit_list[0].index(), state->data_c(),
                state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsOneControlOneTargetGate* copy() const override {
        return new ClsOneControlOneTargetGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }

    void CNOTGateinit(UINT control_qubit_index, UINT target_qubit_index) {
        this->_update_func = CNOT_gate;
        this->_update_func_dm = dm_CNOT_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = CNOT_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = CNOT_gate_mpi;
#endif
        this->_name = "CNOT";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_X_COMMUTE));
        this->_control_qubit_list.push_back(
            ControlQubitInfo(control_qubit_index, 1));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 0, 1, 1, 0;
    }

    void CZGateinit(UINT control_qubit_index, UINT target_qubit_index) {
        this->_update_func = CZ_gate;
        this->_update_func_dm = dm_CZ_gate;
#ifdef _USE_GPU
        this->_update_func_gpu = CZ_gate_host;
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = CZ_gate_mpi;
#endif
        this->_name = "CZ";
        this->_target_qubit_list.push_back(
            TargetQubitInfo(target_qubit_index, FLAG_Z_COMMUTE));
        this->_control_qubit_list.push_back(
            ControlQubitInfo(control_qubit_index, 1));
        this->_gate_property = FLAG_CLIFFORD;
        this->_matrix_element = ComplexMatrix::Zero(2, 2);
        this->_matrix_element << 1, 0, 0, -1;
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.add("name", _name + "Gate");
        pt.add("control_qubit", _control_qubit_list[0].index());
        pt.add("target_qubit", _target_qubit_list[0].index());
        return pt;
    }

    virtual ClsOneControlOneTargetGate* get_inverse(void) const override {
        if (this->_name == "CZ" || this->_name == "CNOT") {
            return this->copy();
        }
        throw NotImplementedException(
            "Inverse of " + this->_name + " gate is not Implemented");
    }
    // 現状CZ,CNOTゲートしかないので、自身がinverseになるが、　そうでないゲートが追加されたときの保険として、　判定をする
};

using QuantumGate_TwoQubit = ClsTwoQubitGate;
using QuantumGate_OneControlOneTarget = ClsOneControlOneTargetGate;
