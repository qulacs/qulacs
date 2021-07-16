#include "pauli_operator.hpp"

#include <boost/dynamic_bitset.hpp>
#include <csim/stat_ops_dm.hpp>
#include <vector>

#ifdef _USE_GPU
#include <gpusim/stat_ops.h>
#endif

#include "state.hpp"
#include "type.hpp"

void MultiQubitPauliOperator::set_bit(
    const UINT pauli_id, const UINT target_index) {
    while (this->_x.size() <= target_index) {
        this->_x.resize(this->_x.size() * 2 + 1);
    }
    this->_z.resize(this->_x.size());
    if (pauli_id == PAULI_ID_X) {
        this->_x.set(target_index);
    } else if (pauli_id == PAULI_ID_Y) {
        this->_x.set(target_index);
        this->_z.set(target_index);
    } else if (pauli_id == PAULI_ID_Z) {
        this->_z.set(target_index);
    }
}

const std::vector<UINT>& MultiQubitPauliOperator::get_pauli_id_list() const {
    return _pauli_id;
}

const std::vector<UINT>& MultiQubitPauliOperator::get_index_list() const {
    return _target_index;
}

void MultiQubitPauliOperator::add_single_Pauli(
    UINT qubit_index, UINT pauli_type) {
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
        } else {
            return dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
                this->_target_index.data(), this->_pauli_id.data(),
                static_cast<UINT>(this->_target_index.size()), state->data_c(),
                state->dim);
        }
    } else if (state->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
        if (state->is_state_vector()) {
            return expectation_value_multi_qubit_Pauli_operator_partial_list_host(
                this->get_index_list().data(), this->get_pauli_id_list().data(),
                (UINT)this->get_index_list().size(), state->data(), state->dim,
                state->get_cuda_stream(), state->device_number);
        } else {
            throw std::runtime_error(
                "Get expectation value for DensityMatrix on GPU is not "
                "supported");
        }
#else
        throw std::invalid_argument("GPU is not supported in this build");
#endif
    } else {
        throw std::invalid_argument("Unsupported device type");
    }
}

CPPCTYPE MultiQubitPauliOperator::get_transition_amplitude(
    const QuantumStateBase* state_bra,
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
        } else {
            throw std::invalid_argument(
                "TransitionAmplitude for density matrix is not implemtend");
        }
    } else if (state_bra->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
        if (state_bra->is_state_vector()) {
            return transition_amplitude_multi_qubit_Pauli_operator_partial_list_host(
                this->get_index_list().data(), this->get_pauli_id_list().data(),
                (UINT)this->get_index_list().size(), state_bra->data(),
                state_ket->data(), state_bra->dim, state_bra->get_cuda_stream(),
                state_bra->device_number);
        } else {
            throw std::runtime_error(
                "Get expectation value for DensityMatrix on GPU is not "
                "supported");
        }
#else
        throw std::invalid_argument("GPU is not supported in this build");
#endif
    } else {
        throw std::invalid_argument("Unsupported device");
    }
}

MultiQubitPauliOperator* MultiQubitPauliOperator::copy() const {
    auto pauli =
        new MultiQubitPauliOperator(this->_target_index, this->_pauli_id);
    return pauli;
}

bool MultiQubitPauliOperator::operator==(
    const MultiQubitPauliOperator& target) const {
    auto x = this->_x;
    auto z = this->_z;
    auto target_x = target.get_x_bits();
    auto target_z = target.get_x_bits();
    if (target_x.size() != this->_x.size()) {
        size_t max_size = std::max(this->_x.size(), target_x.size());
        x.resize(max_size);
        z.resize(max_size);
        target_x.resize(max_size);
        target_z.resize(max_size);
    }
    return x == target_x && z == target_z;
}

MultiQubitPauliOperator MultiQubitPauliOperator::operator*(
    const MultiQubitPauliOperator& target) const {
    auto x = this->_x;
    auto z = this->_z;
    auto target_x = target.get_x_bits();
    auto target_z = target.get_z_bits();
    if (target_x.size() != this->_x.size()) {
        size_t max_size = std::max(x.size(), target_x.size());
        x.resize(max_size);
        z.resize(max_size);
        target_x.resize(max_size);
        target_z.resize(max_size);
    }
    MultiQubitPauliOperator res(x ^ target_x, z ^ target_z);
    return res;
}

MultiQubitPauliOperator& MultiQubitPauliOperator::operator*=(
    const MultiQubitPauliOperator& target) {
    auto target_x = target.get_x_bits();
    auto target_z = target.get_z_bits();
    size_t max_size = std::max(this->_x.size(), target_x.size());
    if (target_x.size() != this->_x.size()) {
        this->_x.resize(max_size);
        this->_z.resize(max_size);
        target_x.resize(max_size);
        target_z.resize(max_size);
    }
    this->_x ^= target_x;
    this->_z ^= target_z;
    _target_index.clear();
    _pauli_id.clear();
    ITYPE i;
    for (i = 0; i < max_size; i++) {
        UINT pauli_id = PAULI_ID_I;
        if (this->_x[i] && !this->_z[i]) {
            pauli_id = PAULI_ID_X;
        } else if (this->_x[i] && this->_z[i]) {
            pauli_id = PAULI_ID_Y;
        } else if (!this->_x[i] && this->_z[i]) {
            pauli_id = PAULI_ID_Z;
        }
        _target_index.push_back(i);
        _pauli_id.push_back(pauli_id);
    }
    return *this;
}

std::string MultiQubitPauliOperator::to_string() const{
    std::string res;
    std::string id;
    ITYPE i;
    for (i = 0; i < _x.size(); i++) {
        if (!_x[i] && !_z[i]) {
            id = "I";
        } else if (_x[i] && !_z[i]) {
            id = "X";
        } else if (_x[i] && _z[i]) {
            id = "Y";
        } else if (!_x[i] && _z[i]) {
            id = "Z";
        }
        if(id!="I"){
            res += id + " " + std::to_string(i) + " ";
        }
    }
    return res;
}