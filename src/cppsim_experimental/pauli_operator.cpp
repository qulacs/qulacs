#include "pauli_operator.hpp"

#include <boost/dynamic_bitset.hpp>
#include <csim/stat_ops_dm.hpp>

#include "type.hpp"

void MultiQubitPauliOperator::set_bit(UINT pauli_id, UINT target_index) {
    while (this->_x.size() <= target_index) {
        this->_x.resize(this->_x.size() * 2 + 1);
        this->_z.resize(this->_z.size() * 2 + 1);
    }
    if (pauli_id == PAULI_ID_X) {
        this->_x.set(target_index);
    }
    else if (pauli_id == PAULI_ID_Y) {
        this->_x.set(target_index);
        this->_z.set(target_index);
    }
    else if (pauli_id == PAULI_ID_Z) {
        this->_z.set(target_index);
    }
}

const std::vector<UINT>& MultiQubitPauliOperator::get_pauli_id_list() const {
    return _pauli_id;
}

const std::vector<UINT>& MultiQubitPauliOperator::get_index_list() const {
    return _target_index;
}

void MultiQubitPauliOperator::add_single_Pauli(UINT qubit_index, UINT pauli_type) {
    if (pauli_type >= 4)
        throw std::invalid_argument("pauli type must be any of 0,1,2,3");
    _target_index.push_back(qubit_index);
    _pauli_id.push_back(pauli_type);
    set_bit(pauli_type, qubit_index);
}

CPPCTYPE MultiQubitPauliOperator::get_expectation_value(
    const QuantumStateBase* state) const {
    if (state->get_device_type() == DEVICE_CPU) {
        if (state->is_state_vector()) {
            return expectation_value_multi_qubit_Pauli_operator_partial_list(
                this->_target_index.data(), this->_pauli_id.data(),
                static_cast<UINT>(this->_target_index.size()), state->data_c(),
                state->dim);
        }
        else {
            return dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
                this->_target_index.data(), this->_pauli_id.data(),
                static_cast<UINT>(this->_target_index.size()), state->data_c(),
                state->dim);
        }
    }
    else if (state->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
        if (state->is_state_vector()) {
            return expectation_value_multi_qubit_Pauli_operator_partial_list_host(
                this->get_index_list().data(),
                this->get_pauli_id_list().data(),
                (UINT)this->get_index_list().size(), state->data(),
                state->dim, state->get_cuda_stream(), state->device_number);
        }
        else {
            throw std::runtime_error(
                "Get expectation value for DensityMatrix on GPU is not "
                "supported");
        }
#else
        throw std::invalid_argument("GPU is not supported in this build");
#endif
    }
    else {
        throw std::invalid_argument("Unsupported device type");
    }
}

CPPCTYPE MultiQubitPauliOperator::get_transition_amplitude(const QuantumStateBase* state_bra,
    const QuantumStateBase* state_ket) const {
    if (state_bra->get_device_type() != state_ket->get_device_type())
        throw std::invalid_argument("Device type is different");
    if (state_bra->is_state_vector() != state_ket->is_state_vector())
        throw std::invalid_argument("is_state_vector is not matched");
    if (state_bra->dim != state_ket->dim)
        throw std::invalid_argument("state_bra->dim != state_ket->dim");

    if (state_bra->get_device_type() == DEVICE_CPU) {
        if (state_bra->is_state_vector()) {
            return transition_amplitude_multi_qubit_Pauli_operator_partial_list(
                this->_target_index.data(), this->_pauli_id.data(),
                (UINT)this->_target_index.size(), state_bra->data_c(),
                state_ket->data_c(), state_bra->dim);
        }
        else {
            throw std::invalid_argument(
                "TransitionAmplitude for density matrix is not implemtend");
        }
    }
    else if (state_bra->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
        if (state_bra->is_state_vector()) {
            return transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(
                this->get_index_list().data(),
                this->get_pauli_id_list().data(),
                (UINT)this->get_index_list().size(), state_bra->data(),
                state_ket->data(), state_bra->dim,
                state_bra->get_cuda_stream(), state_bra->device_number);
        }
        else {
            throw std::runtime_error(
                "Get expectation value for DensityMatrix on GPU is not "
                "supported");
        }
#else
        throw std::invalid_argument("GPU is not supported in this build");
#endif
    }
    else {
        throw std::invalid_argument("Unsupported device");
    }
}
