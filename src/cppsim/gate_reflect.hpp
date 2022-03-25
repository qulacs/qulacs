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
 * �ʎq��Ԃ��A�ʂ̗ʎq��Ԃɑ΂��Ĕ��˂���Q�[�g�̃N���X
 */
class ClsStateReflectionGate : public QuantumGateBase {
private:
    QuantumStateBase* reflection_state;

public:
    explicit ClsStateReflectionGate(const QuantumStateBase* _reflection_state) {
        reflection_state = _reflection_state->copy();
        UINT qubit_count = _reflection_state->qubit_count;
        for (UINT qubit_index = 0; qubit_index < qubit_count; ++qubit_index) {
            this->_target_qubit_list.push_back(TargetQubitInfo(qubit_index, 0));
        }
        this->_name = "Reflection";
    };
    virtual ~ClsStateReflectionGate() { delete reflection_state; }

    /**
     * \~japanese-en �ʎq��Ԃ��X�V����
     *
     * @param state �X�V����ʎq���
     */
    virtual void update_quantum_state(QuantumStateBase* state) override {
        if (state->is_state_vector()) {
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
     * ���g�̃f�B�[�v�R�s�[�𐶐�����
     *
     * @return ���g�̃f�B�[�v�R�s�[
     */
    virtual QuantumGateBase* copy() const override {
        return new ClsStateReflectionGate(this->reflection_state);
    };
    /**
     * \~japanese-en ���g�̃Q�[�g�s����Z�b�g����
     *
     * @param matrix �s����Z�b�g����ϐ��̎Q��
     */
    virtual void set_matrix(ComplexMatrix&) const override {
        throw NotImplementedException(
            "ReflectionGate::set_matrix is not implemented");
    }
};
