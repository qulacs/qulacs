
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

#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"

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
