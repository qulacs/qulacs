
/**
 * @file pauli_operator.hpp
 * @brief Definition and basic functions for MultiPauliTerm
 */

#pragma once

#include <boost/dynamic_bitset.hpp>
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
    boost::dynamic_bitset<> _z;
    boost::dynamic_bitset<> _x;

    void set_bit(UINT pauli_id, UINT target_index) {
        while (_x.size() <= target_index) {
            _x.resize(_x.size() * 2 + 1);
            _z.resize(_z.size() * 2 + 1);
        }
        if (pauli_id == PAULI_ID_X) {
            _x.set(target_index);
        } else if (pauli_id == PAULI_ID_Y) {
            _x.set(target_index);
            _z.set(target_index);
        } else if (pauli_id == PAULI_ID_Z) {
            _z.set(target_index);
        }
    }

public:
    virtual const std::vector<UINT>& get_pauli_id_list() const {
        return _pauli_id;
    }
    virtual const std::vector<UINT>& get_index_list() const {
        return _target_index;
    }
    virtual const boost::dynamic_bitset<>& get_x_bits() const { return _x; }
    virtual const boost::dynamic_bitset<>& get_z_bits() const { return _z; }
    MultiQubitPauliOperator(){};
    MultiQubitPauliOperator(const std::vector<UINT>& target_qubit_index_list,
        const std::vector<UINT>& pauli_id_list)
        : _target_index(target_qubit_index_list), _pauli_id(pauli_id_list) {
        ITYPE i;
        for (i = 0; i < pauli_id_list.size(); i++) {
            set_bit(pauli_id_list[i], target_qubit_index_list[i]);
        }
    };

    explicit MultiQubitPauliOperator(std::string pauli_string) {
        std::string pattern = "([IXYZ])\\s*([0-9]+)\\s*";
        std::regex re(pattern);
        std::cmatch result;
        while (std::regex_search(pauli_string.c_str(), result, re)) {
            std::string pauli = result[1].str();
            UINT index = (UINT)std::stoul(result[2].str());
            _target_index.push_back(index);
            UINT pauli_id;
            if (pauli == "I")
                pauli_id = PAULI_ID_I;
            else if (pauli == "X")
                pauli_id = PAULI_ID_X;
            else if (pauli == "Y")
                pauli_id = PAULI_ID_Y;
            else if (pauli == "Z")
                pauli_id = PAULI_ID_Z;
            else
                assert(false && "Error in regex");
            _pauli_id.push_back(pauli_id);
            set_bit(pauli_id, index);
            pauli_string = result.suffix();
        }
        assert(_target_index.size() == _pauli_id.size());
    }

    MultiQubitPauliOperator(
        const boost::dynamic_bitset<>& x, const boost::dynamic_bitset<>& z)
        : _x(x), _z(z) {
        ITYPE index;
#pragma omp parallel for
        for (index = 0; index < _x.size(); index++) {
            UINT pauli_id;
            if (!_x[index] && !_z[index])
                pauli_id = PAULI_ID_I;
            else if (!_x[index] && _z[index])
                pauli_id = PAULI_ID_Z;
            else if (_x[index] && !_z[index])
                pauli_id = PAULI_ID_X;
            else if (_x[index] && _z[index])
                pauli_id = PAULI_ID_Y;
            _target_index.push_back(index);
            _pauli_id.push_back(pauli_id);
        }
    };

    virtual ~MultiQubitPauliOperator(){};
    virtual void add_single_Pauli(UINT qubit_index, UINT pauli_type) {
        if (pauli_type >= 4)
            throw std::invalid_argument("pauli type must be any of 0,1,2,3");
        _target_index.push_back(qubit_index);
        _pauli_id.push_back(pauli_type);
        set_bit(pauli_type, qubit_index);
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

    MultiQubitPauliOperator* copy() const {
        auto pauli = new MultiQubitPauliOperator(_target_index, _pauli_id);
        return pauli;
    }

    bool operator==(const MultiQubitPauliOperator& target) const {
        auto x = _x;
        auto z = _z;
        auto target_x = target.get_x_bits();
        auto target_z = target.get_x_bits();
        if (target_x.size() != _x.size()) {
            size_t max_size = std::max(_x.size(), target_x.size());
            x.resize(max_size);
            z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        return x == target_x && z == target_z;
    }

    MultiQubitPauliOperator operator*(
        const MultiQubitPauliOperator& target) const {
        auto x = _x;
        auto z = _z;
        auto target_x = target.get_x_bits();
        auto target_z = target.get_x_bits();
        if (target_x.size() != _x.size()) {
            size_t max_size = std::max(_x.size(), target_x.size());
            x.resize(max_size);
            z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        MultiQubitPauliOperator res(x ^ target_x, z ^ target_z);
        return res;
    }

    MultiQubitPauliOperator& operator*=(const MultiQubitPauliOperator& target) {
        auto target_x = target.get_x_bits();
        auto target_z = target.get_z_bits();
        size_t max_size = std::max(_x.size(), target_x.size());
        if (target_x.size() != _x.size()) {
            _x.resize(max_size);
            _z.resize(max_size);
            target_x.resize(max_size);
            target_z.resize(max_size);
        }
        _x ^= target_x;
        _z ^= target_z;
        _target_index.clear();
        _pauli_id.clear();
        ITYPE i;
#pragma omp parallel for
        for (i = 0; i < max_size; i++) {
            UINT pauli_id = PAULI_ID_I;
            if (_x[i] && !_z[i]) {
                pauli_id = PAULI_ID_X;
            } else if (_x[i] && _z[i]) {
                pauli_id = PAULI_ID_Y;
            } else if (!_x[i] && _z[i]) {
                pauli_id = PAULI_ID_Z;
            }
            _target_index.push_back(i);
            _pauli_id.push_back(pauli_id);
        }
        return *this;
    }
};

using PauliOperator = MultiQubitPauliOperator;

class DllExport Observable {
private:
    std::vector<MultiQubitPauliOperator> _pauli_terms;
    std::vector<CPPCTYPE> _coef_list;

    CPPCTYPE culc_coef(const MultiQubitPauliOperator& a,
        const MultiQubitPauliOperator& b) const {
        auto x_a = a.get_x_bits();
        auto z_a = a.get_z_bits();
        auto x_b = b.get_x_bits();
        auto z_b = b.get_z_bits();
        size_t max_size = std::max(x_a.size(), x_b.size());
        if (x_a.size() != x_b.size()) {
            x_a.resize(max_size);
            z_a.resize(max_size);
            x_b.resize(max_size);
            z_b.resize(max_size);
        }
        CPPCTYPE res = 1.0;
        CPPCTYPE I = 1.0i;
        ITYPE i;
#pragma omp parallel for
        for (i = 0; i < x_a.size(); i++) {
            if (x_a[i] && !z_a[i]) {  // X
                if (!x_b[i] && z_b[i]) {
                    res = res * -I;
                } else if (x_b[i] && z_b[i]) {
                    res = res * I;
                }
            } else if (!x_a[i] && z_a[i]) {  // Z
                if (x_b[i] && !z_b[i]) {     // X
                    res = res * -I;
                } else if (x_b[i] && z_b[i]) {  // Y
                    res = res * I;
                }
            } else if (x_a[i] && z_a[i]) {  // Y
                if (x_b[i] && !z_b[i]) {    // X
                    res = res * I;
                } else if (!x_b[i] && z_b[i]) {  // Z
                    res = res * I;
                }
            }
        }
        return res;
    }

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
        ITYPE index;
        for (index = 0; index < _pauli_terms.size(); ++index) {
            sum += _coef_list.at(index) *
                   _pauli_terms.at(index).get_expectation_value(state);
        }
        return sum;
    }
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const {
        CPPCTYPE sum = 0;
        ITYPE index;
        for (index = 0; index < _pauli_terms.size(); ++index) {
            sum += _coef_list.at(index) *
                   _pauli_terms.at(index).get_transition_amplitude(
                       state_bra, state_ket);
        }
        return sum;
    }

    virtual Observable* copy() const {
        auto res = new Observable();
        ITYPE i;
#pragma omp parallel for
        for (i = 0; i < _coef_list.size(); i++) {
            res->add_term(_coef_list[i], *_pauli_terms[i].copy());
        }
        return res;
    }

    Observable operator+(const Observable& target) const {
        auto res = this->copy();
        *res += target;
        return *res;
    }

    Observable& operator+=(const Observable& target) {
        ITYPE i, j;
#pragma omp parallel for
        for (j = 0; j < target.get_term_count(); j++) {
            auto term = target.get_term(j);
            bool flag = true;
            for (int i = 0; i < _pauli_terms.size(); i++) {
                if (_pauli_terms[i] == term.second) {
                    _coef_list[i] += term.first;
                    flag = false;
                }
            }
            if (flag) {
                this->add_term(term.first, term.second);
            }
        }
        return *this;
    }

    Observable operator-(const Observable& target) const {
        auto res = *this->copy();
        res -= target;
        return res;
    }

    Observable& operator-=(const Observable& target) {
        ITYPE i, j;
#pragma omp parallel for
        for (j = 0; j < target.get_term_count(); j++) {
            auto term = target.get_term(j);
            bool flag = true;
            for (int i = 0; i < _pauli_terms.size(); i++) {
                if (_pauli_terms[i] == term.second) {
                    _coef_list[i] -= term.first;
                    flag = false;
                }
            }
            if (flag) {
                this->add_term(-term.first, term.second);
            }
        }
        return *this;
    }

    Observable operator*(const Observable& target) const {
        Observable res;
        ITYPE i, j;
#pragma omp parallel for
        for (i = 0; i < _pauli_terms.size(); i++) {
            for (j = 0; j < target.get_term_count(); j++) {
                Observable tmp;
                auto term = target.get_term(j);
                CPPCTYPE bits_coef = culc_coef(_pauli_terms[i], term.second);
                tmp.add_term(_coef_list[i] * term.first * bits_coef,
                    _pauli_terms[i] * term.second);
                res += tmp;
            }
        }
        return res;
    }

    Observable operator*(const CPPCTYPE& target) const {
        auto res = *this->copy();
        res *= target;
        return res;
    }

    Observable& operator*=(const Observable& target) {
        auto tmp = *this->copy() * target;
        _coef_list.clear();
        _pauli_terms.clear();
        ITYPE i;
#pragma omp parallel for
        for (i = 0; i < tmp.get_term_count(); i++) {
            auto term = tmp.get_term(i);
            this->add_term(term.first, term.second);
        }
        return *this;
    }

    Observable& operator*=(const CPPCTYPE& target) {
        ITYPE i;
#pragma omp parallel for
        for (int i = 0; i < _coef_list.size(); i++) {
            _coef_list[i] *= target;
        }
        return *this;
    }
};
