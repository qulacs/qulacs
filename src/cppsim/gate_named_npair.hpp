#pragma once

#include <csim/update_ops.hpp>

#include "gate.hpp"
#include "state.hpp"
#include "utility.hpp"

/**
 * \~japanese-en (2n)量子ビットを対象とする回転角固定のゲートのクラス
 */
class ClsNpairQubitGate : public QuantumGateBase {
protected:
    using UpdateFunc = void (*)(UINT, UINT, UINT, CTYPE*, ITYPE);
    using UpdateFuncGpu = void (*)(UINT, UINT, UINT, void*, ITYPE, void*, UINT);
    using UpdateFuncMpi = void (*)(UINT, UINT, UINT, CTYPE*, ITYPE, UINT);
    UpdateFunc _update_func;
    UpdateFunc _update_func_dm;
    UpdateFuncGpu _update_func_gpu;
    UpdateFuncMpi _update_func_mpi;
    SparseComplexMatrix _matrix_element;
    UINT _block_size;

public:
    explicit ClsNpairQubitGate(){};
    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                throw NotImplementedException(
                    this->_name + " gate is not Implemented");
                // _update_func_gpu(this->_target_qubit_list[0].index(),
                //     this->_target_qubit_list[1].index(), this->_block_size,
                //     state->data(), state->dim, state->get_cuda_stream(),
                //     state->device_number);
            } else
#endif
#ifdef _USE_MPI
                if (state->outer_qc > 0) {
                _update_func_mpi(this->_target_qubit_list[0].index(),
                    this->_target_qubit_list[_block_size].index(),
                    this->_block_size, state->data_c(), state->dim,
                    state->inner_qc);
            } else
#endif
            {
                _update_func(this->_target_qubit_list[0].index(),
                    this->_target_qubit_list[_block_size].index(),
                    this->_block_size, state->data_c(), state->dim);
            }
        } else {
            throw NotImplementedException(
                this->_name + " gate is not Implemented");
            // _update_func_dm(this->_target_qubit_list[0].index(),
            //     this->_target_qubit_list[1].index(), this->_block_size,
            //     state->data_c(), state->dim);
        }
    };
    /**
     * \~japanese-en 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsNpairQubitGate* copy() const override {
        return new ClsNpairQubitGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        matrix = this->_matrix_element;
    }

    void FusedSWAPGateinit(
        UINT target_qubit_index1, UINT target_qubit_index2, UINT block_size) {
        this->_update_func = FusedSWAP_gate;
        this->_update_func_dm = nullptr;  // not supported
#ifdef _USE_GPU
        this->_update_func_gpu = nullptr;  // not supported
#endif
#ifdef _USE_MPI
        this->_update_func_mpi = FusedSWAP_gate_mpi;
#endif
        this->_name = "FusedSWAP";
        // 以下の順序でtarget_qubit_listに追加
        // [target_qubit_index1, target_qubit_index1+1, ...,
        // target_qubit_index1+(num_qubits-1),
        //  target_qubit_index2, target_qubit_index2+1, ...,
        //  target_qubit_index2+(num_qubits-1)]
        for (UINT i = 0; i < block_size; ++i)
            this->_target_qubit_list.push_back(
                TargetQubitInfo(target_qubit_index1 + i, 0));
        for (UINT i = 0; i < block_size; ++i)
            this->_target_qubit_list.push_back(
                TargetQubitInfo(target_qubit_index2 + i, 0));
        this->_block_size = block_size;
        this->_gate_property = FLAG_CLIFFORD;
        // matrix生成
        const ITYPE pow2_nq = 1ULL << block_size;
        const ITYPE pow2_2nq = 1ULL << (block_size * 2);
        this->_matrix_element = SparseComplexMatrix(pow2_2nq, pow2_2nq);
        this->_matrix_element.reserve(pow2_2nq);
        for (ITYPE i = 0; i < pow2_nq; i++) {
            for (ITYPE j = 0; j < pow2_nq; j++) {
                this->_matrix_element.insert(i * pow2_nq + j, i + j * pow2_nq) =
                    1;
            }
        }
    }

    /**
     * \~japanese-en ptreeに変換する
     *
     * @return ptree
     */
    virtual boost::property_tree::ptree to_ptree() const {
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

    virtual ClsNpairQubitGate* get_inverse(void) const override {
        if (this->_name == "FusedSWAP") {
            return this->copy();
        }
        throw NotImplementedException(
            "Inverse of " + this->_name + " gate is not Implemented");
    }
    // 現状FusedSWAPゲートしかないので、自身がget_inverseになるが、　そうでないゲートが追加されたときの保険として、　判定をする
};

using QuantumGate_NpairQubit = ClsNpairQubitGate;
