#include "observable.hpp"

#include <numeric>

#ifdef _USE_GPU
#include <gpusim/stat_ops.h>
#endif

#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"

CPPCTYPE Observable::calc_coef(
    const MultiQubitPauliOperator& a, const MultiQubitPauliOperator& b) const {
    auto x_a = a.get_x_bits();
    auto z_a = a.get_z_bits();
    auto x_b = b.get_x_bits();
    auto z_b = b.get_z_bits();
    const size_t max_size = std::max(x_a.size(), x_b.size());
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
                res *= -I;
            } else if (x_b[i] && z_b[i]) {
                res *= I;
            }
        } else if (!x_a[i] && z_a[i]) {  // Z
            if (x_b[i] && !z_b[i]) {     // X
                res *= -I;
            } else if (x_b[i] && z_b[i]) {  // Y
                res *= I;
            }
        } else if (x_a[i] && z_a[i]) {  // Y
            if (x_b[i] && !z_b[i]) {    // X
                res *= I;
            } else if (!x_b[i] && z_b[i]) {  // Z
                res *= I;
            }
        }
    }
    return res;
}

std::pair<CPPCTYPE, MultiQubitPauliOperator> Observable::get_term(
    const UINT index) const {
    return std::make_pair(
        this->_coef_list.at(index), this->_pauli_terms.at(index));
}

void Observable::add_term(const CPPCTYPE coef, MultiQubitPauliOperator op) {
    this->_coef_list.push_back(coef);
    this->_pauli_terms.push_back(op);
}

void Observable::add_term(const CPPCTYPE coef, std::string s) {
    this->_coef_list.push_back(coef);
    this->_pauli_terms.push_back(MultiQubitPauliOperator(s));
}

void Observable::remove_term(UINT index) {
    this->_coef_list.erase(this->_coef_list.begin() + index);
    this->_pauli_terms.erase(this->_pauli_terms.begin() + index);
}

CPPCTYPE Observable::get_expectation_value(
    const QuantumStateBase* state) const {
    CPPCTYPE sum = 0;
    for (ITYPE index = 0; index < this->_pauli_terms.size(); ++index) {
        sum += this->_coef_list.at(index) *
               this->_pauli_terms.at(index).get_expectation_value(state);
    }
    return sum;
}

CPPCTYPE Observable::get_transition_amplitude(const QuantumStateBase* state_bra,
    const QuantumStateBase* state_ket) const {
    CPPCTYPE sum = 0;
    for (ITYPE index = 0; index < this->_pauli_terms.size(); ++index) {
        sum += this->_coef_list.at(index) *
               this->_pauli_terms.at(index).get_transition_amplitude(
                   state_bra, state_ket);
    }
    return sum;
}

Observable* Observable::copy() const {
    auto res = new Observable();
    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < this->_coef_list.size(); i++) {
        res->add_term(this->_coef_list[i], *this->_pauli_terms[i].copy());
    }
    return res;
}

Observable Observable::operator+(const Observable& target) const {
    auto res = this->copy();
    *res += target;
    return *res;
}

Observable& Observable::operator+=(const Observable& target) {
    ITYPE i, j;
#pragma omp parallel for
    for (j = 0; j < target.get_term_count(); j++) {
        auto term = target.get_term(j);
        bool flag = true;
        for (int i = 0; i < this->_pauli_terms.size(); i++) {
            if (this->_pauli_terms[i] == term.second) {
                this->_coef_list[i] += term.first;
                flag = false;
            }
        }
        if (flag) {
            this->add_term(term.first, term.second);
        }
    }
    return *this;
}

Observable Observable::operator-(const Observable& target) const {
    auto res = *this->copy();
    res -= target;
    return res;
}

Observable& Observable::operator-=(const Observable& target) {
    ITYPE i, j;
#pragma omp parallel for
    for (j = 0; j < target.get_term_count(); j++) {
        auto term = target.get_term(j);
        bool flag = true;
        for (int i = 0; i < this->_pauli_terms.size(); i++) {
            if (this->_pauli_terms[i] == term.second) {
                this->_coef_list[i] -= term.first;
                flag = false;
            }
        }
        if (flag) {
            this->add_term(-term.first, term.second);
        }
    }
    return *this;
}

Observable Observable::operator*(const Observable& target) const {
    Observable res;
    ITYPE i, j;
#pragma omp parallel for
    for (i = 0; i < this->_pauli_terms.size(); i++) {
        for (j = 0; j < target.get_term_count(); j++) {
            Observable tmp;
            auto term = target.get_term(j);
            CPPCTYPE bits_coef = calc_coef(this->_pauli_terms[i], term.second);
            tmp.add_term(this->_coef_list[i] * term.first * bits_coef,
                this->_pauli_terms[i] * term.second);
            res += tmp;
        }
    }
    return res;
}

Observable Observable::operator*(const CPPCTYPE& target) const {
    auto res = *this->copy();
    res *= target;
    return res;
}

Observable& Observable::operator*=(const Observable& target) {
    auto tmp = *this->copy() * target;
    this->_coef_list.clear();
    this->_pauli_terms.clear();
    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < tmp.get_term_count(); i++) {
        auto term = tmp.get_term(i);
        this->add_term(term.first, term.second);
    }
    return *this;
}

Observable& Observable::operator*=(const CPPCTYPE& target) {
    ITYPE i;
#pragma omp parallel for
    for (int i = 0; i < this->_coef_list.size(); i++) {
        this->_coef_list[i] *= target;
    }
    return *this;
}