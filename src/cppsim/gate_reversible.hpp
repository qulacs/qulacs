#pragma once

#include <cmath>
#include <csim/update_ops.hpp>
#include <csim/update_ops_cpp.hpp>

#include "exception.hpp"
#include "gate.hpp"
#include "state.hpp"

#ifdef _USE_GPU
#include <gpusim/update_ops_cuda.h>
#endif
#include <iostream>

/**
 * \~japanese-en 可逆古典回路のを表すクラス
 */
class ClsReversibleBooleanGate : public QuantumGateBase {
private:
    std::function<ITYPE(ITYPE, ITYPE)> function_ptr;

public:
    ClsReversibleBooleanGate(std::vector<UINT> target_qubit_index_list,
        std::function<ITYPE(ITYPE, ITYPE)> _function_ptr)
        : function_ptr(_function_ptr) {
        for (auto val : target_qubit_index_list) {
            this->_target_qubit_list.push_back(TargetQubitInfo(val, 0));
        }
        this->_name = "ReversibleBoolean";
    };

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        std::vector<UINT> target_index;
        std::transform(this->_target_qubit_list.cbegin(),
            this->_target_qubit_list.cend(), std::back_inserter(target_index),
            [](auto value) { return value.index(); });
        if (state->is_state_vector()) {
#ifdef _USE_GPU
            if (state->get_device_name() == "gpu") {
                throw NotImplementedException("Not Implemented");
                // reversible_boolean_gate_gpu(target_index.data(),
                // target_index.size(), function_ptr, state->data_c(),
                // state->dim);
            } else {
                reversible_boolean_gate(target_index.data(),
                    (UINT)target_index.size(), function_ptr, state->data_c(),
                    state->dim);
            }
#else
            reversible_boolean_gate(target_index.data(),
                (UINT)target_index.size(), function_ptr, state->data_c(),
                state->dim);
#endif
        } else {
            throw NotImplementedException("not implemented");
        }
    };
    /**
     * \~japanese-en
     * 自身のディープコピーを生成する
     *
     * @return 自身のディープコピー
     */
    virtual ClsReversibleBooleanGate* copy() const override {
        return new ClsReversibleBooleanGate(*this);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix& matrix) const override {
        ITYPE matrix_dim = 1ULL << this->_target_qubit_list.size();
        matrix = ComplexMatrix::Zero(matrix_dim, matrix_dim);
        for (ITYPE index = 0; index < matrix_dim; ++index) {
            ITYPE target_index = function_ptr(index, matrix_dim);
            matrix(target_index, index) = 1;
        }
    }
};
