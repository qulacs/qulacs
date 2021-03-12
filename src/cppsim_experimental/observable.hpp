
/**
 * @file pauli_operator.hpp
 * @brief Definition and basic functions for MultiPauliTerm
 */

#pragma once

#include <cassert>
#include <csim/stat_ops.hpp>
#include <csim/stat_ops_dm.hpp>
#include <iostream>
#include <regex>
#include <vector>

#ifdef _USE_GPU
#include <gpusim/stat_ops.h>
#endif

#include "state.hpp"
#include "type.hpp"

enum {
    PAULI_ID_I = 0,
    PAULI_ID_X = 1,
    PAULI_ID_Y = 2,
    PAULI_ID_Z = 3,
};

class DllExport MultiQubitPauliOperator {
private:
    std::vector<UINT> _target_index;
    std::vector<UINT> _pauli_id;

public:
    virtual const std::vector<UINT>& get_pauli_id_list() const {
        return _pauli_id;
    }
    virtual const std::vector<UINT>& get_index_list() const {
        return _target_index;
    }

    MultiQubitPauliOperator(){};
    MultiQubitPauliOperator(const std::vector<UINT>& target_qubit_index_list,
        const std::vector<UINT>& pauli_id_list)
        : _target_index(target_qubit_index_list), _pauli_id(pauli_id_list){};

    explicit MultiQubitPauliOperator(std::string pauli_string) {
        std::string pattern = "([IXYZ])\\s*([0-9]+)\\s*";
        std::regex re(pattern);
        std::cmatch result;
        while (std::regex_search(pauli_string.c_str(), result, re)) {
            std::string pauli = result[1].str();
            UINT index = (UINT)std::stoul(result[2].str());
            _target_index.push_back(index);
            if (pauli == "I")
                _pauli_id.push_back(PAULI_ID_I);
            else if (pauli == "X")
                _pauli_id.push_back(PAULI_ID_X);
            else if (pauli == "Y")
                _pauli_id.push_back(PAULI_ID_Y);
            else if (pauli == "Z")
                _pauli_id.push_back(PAULI_ID_Z);
            else
                assert(false && "Error in regex");
            pauli_string = result.suffix();
        }
        assert(_target_index.size() == _pauli_id.size());
    }

    virtual ~MultiQubitPauliOperator(){};
    virtual void add_single_Pauli(UINT qubit_index, UINT pauli_type) {
        if (pauli_type >= 4)
            throw std::invalid_argument("pauli type must be any of 0,1,2,3");
        _target_index.push_back(qubit_index);
        _pauli_id.push_back(pauli_type);
    }

    virtual CPPCTYPE get_expectation_value(
        const QuantumStateBase* state) const {
        if (state->get_device_type() == DEVICE_CPU) {
            if (state->is_state_vector()) {
                return expectation_value_multi_qubit_Pauli_operator_partial_list(
                    this->_target_index.data(), this->_pauli_id.data(),
                    (UINT)this->_target_index.size(), state->data_c(),
                    state->dim);
            } else {
                return dm_expectation_value_multi_qubit_Pauli_operator_partial_list(
                    this->_target_index.data(), this->_pauli_id.data(),
                    (UINT)this->_target_index.size(), state->data_c(),
                    state->dim);
            }
        } else if (state->get_device_type() == DEVICE_GPU) {
#ifdef _USE_GPU
            if (state->is_state_vector()) {
                return expectation_value_multi_qubit_Pauli_operator_partial_list_host(
                    this->get_index_list().data(),
                    this->get_pauli_id_list().data(),
                    (UINT)this->get_index_list().size(), state->data(),
                    state->dim, state->get_cuda_stream(), state->device_number);
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
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
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
                    this->get_index_list().data(),
                    this->get_pauli_id_list().data(),
                    (UINT)this->get_index_list().size(), state_bra->data(),
                    state_ket->data(), state_bra->dim,
                    state_bra->get_cuda_stream(), state_bra->device_number);
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
};

using PauliOperator = MultiQubitPauliOperator;

class DllExport Observable {
private:
    std::vector<MultiQubitPauliOperator> _pauli_terms;
    std::vector<CPPCTYPE> _coef_list;

public:
    Observable(){};
    virtual UINT get_term_count() const { return (UINT)_pauli_terms.size(); }
    virtual std::pair<CPPCTYPE, MultiQubitPauliOperator> get_term(
        UINT index) const {
        return std::make_pair(_coef_list.at(index), _pauli_terms.at(index));
    }
    virtual void add_term(CPPCTYPE coef, MultiQubitPauliOperator op) {
        _coef_list.push_back(coef);
        _pauli_terms.push_back(op);
    }
    virtual void add_term(CPPCTYPE coef, std::string s) {
        _coef_list.push_back(coef);
        _pauli_terms.push_back(MultiQubitPauliOperator(s));
    }
    virtual void remove_term(UINT index) {
        _coef_list.erase(_coef_list.begin() + index);
        _pauli_terms.erase(_pauli_terms.begin() + index);
    }

    virtual CPPCTYPE get_expectation_value(
        const QuantumStateBase* state) const {
        CPPCTYPE sum = 0;
        for (UINT index = 0; index < _pauli_terms.size(); ++index) {
            sum += _coef_list.at(index) *
                   _pauli_terms.at(index).get_expectation_value(state);
        }
        return sum;
    }
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const {
        CPPCTYPE sum = 0;
        for (UINT index = 0; index < _pauli_terms.size(); ++index) {
            sum += _coef_list.at(index) *
                   _pauli_terms.at(index).get_transition_amplitude(
                       state_bra, state_ket);
        }
        return sum;
    }
};
