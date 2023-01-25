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
 * \~japanese-en
 * 量子状態を、別の量子状態に対して反射するゲートのクラス
 */
class ClsStateReflectionGate : public QuantumGateBase {
private:
    QuantumState* reflection_state;

public:
    explicit ClsStateReflectionGate(const QuantumState* _reflection_state) {
        reflection_state = _reflection_state->copy();
        UINT qubit_count = _reflection_state->qubit_count;
        for (UINT qubit_index = 0; qubit_index < qubit_count; ++qubit_index) {
            this->_target_qubit_list.push_back(TargetQubitInfo(qubit_index, 0));
        }
        this->_name = "Reflection";
    };
    virtual ~ClsStateReflectionGate() { delete reflection_state; }

    /**
     * \~japanese-en 量子状態を更新する
     *
     * @param state 更新する量子状態
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
            if (state->qubit_count != reflection_state->qubit_count) {
                throw InvalidQubitCountException(
                    "ClsStateReflectionGate::update_quantumstate("
                    "QuantumStateBase*): qubit count must be equal to "
                    "reflection_state's");
            }
#ifdef _USE_GPU
            if (state->get_device_name() !=
                reflection_state->get_device_name()) {
                throw NotImplementedException(
                    "Quantum state on CPU (GPU) cannot be reflected using "
                    "quantum state on GPU (CPU)");
            }
            if (state->get_device_name() == "gpu") {
                std::stringstream error_message_stream;
                error_message_stream << "Not Implemented";
                throw NotImplementedException(error_message_stream.str());
                // reversible_boolean_gate_gpu(target_index.data(),
                // target_index.size(), function_ptr, state->data_c(),
                // state->dim);
            } else {
                reflection_gate(
                    reflection_state->data_c(), state->data_c(), state->dim);
            }
#else
            reflection_gate(
                reflection_state->data_c(), state->data_c(), state->dim);
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
    virtual ClsStateReflectionGate* copy() const override {
        return new ClsStateReflectionGate(this->reflection_state);
    };
    /**
     * \~japanese-en 自身のゲート行列をセットする ことになっているが、実際はnot
     * implemented
     *
     * @param matrix 行列をセットする変数の参照
     */
    virtual void set_matrix(ComplexMatrix&) const override {
        throw NotImplementedException(
            "ReflectionGate::set_matrix is not implemented");
    }
    /**
     * \~japanese-en ptreeに変換する
     *
     * @param ptree ptree
     */
    virtual boost::property_tree::ptree to_ptree() const override {
        boost::property_tree::ptree pt;
        pt.put("name", "StateReflectionGate");
        pt.put_child("reflection_state", reflection_state->to_ptree());
        return pt;
    }
};
