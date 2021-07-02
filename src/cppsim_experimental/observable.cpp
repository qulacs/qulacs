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
    for (i = 0; i < x_a.size(); i++) {
        if (x_a[i] && !z_a[i]) {            // a = X
            if (!x_b[i] && z_b[i]) {        // b = Z
                res *= -I;                  // XZ = -iY
            } else if (x_b[i] && z_b[i]) {  // b = Y
                res *= I;                   // XY = iZ
            }
        } else if (!x_a[i] && z_a[i]) {     // a = Z
            if (x_b[i] && !z_b[i]) {        // b = X
                res *= I;                   // ZX = iY
            } else if (x_b[i] && z_b[i]) {  // b = Y
                res *= -I;                  // ZY = -iX
            }
        } else if (x_a[i] && z_a[i]) {       // a = Y
            if (x_b[i] && !z_b[i]) {         // b = X
                res *= -I;                   // YX = -iZ
            } else if (!x_b[i] && z_b[i]) {  // b = Z
                res *= I;                    // YZ = iX
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
    Observable* res = new Observable();
    ITYPE i;
    for (i = 0; i < this->_coef_list.size(); i++) {
        res->add_term(this->_coef_list[i], *this->_pauli_terms[i].copy());
    }
    return res;
}

Observable Observable::operator+(const Observable& target) const {
    Observable res = *(this->copy());
    res += target;
    return res;
}

Observable& Observable::operator+=(const Observable& target) {
    ITYPE i;
    std::unordered_map<std::string, ITYPE> u_map;
    for (i = 0; i < this->_pauli_terms.size(); i++) {
        u_map[_pauli_terms[i].to_string()] = i;
    }

    for (i = 0; i < target.get_term_count(); i++) {
        auto term = target.get_term(i);
        if (u_map.find(term.second.to_string()) != u_map.end()) {
            ITYPE id = u_map[term.second.to_string()];
            this->_coef_list[id] += term.first;
        } else {
            this->add_term(term.first, term.second);
        }
    }
    return *this;
}

Observable Observable::operator-(const Observable& target) const {
    Observable res = *(this->copy());
    res -= target;
    return res;
}

Observable& Observable::operator-=(const Observable& target) {
    ITYPE i;
    std::unordered_map<std::string, ITYPE> u_map;
    for (i = 0; i < this->_pauli_terms.size(); i++) {
        u_map[_pauli_terms[i].to_string()] = i;
    }

    for (i = 0; i < target.get_term_count(); i++) {
        auto term = target.get_term(i);
        if (u_map.find(term.second.to_string()) != u_map.end()) {
            ITYPE id = u_map[term.second.to_string()];
            this->_coef_list[id] -= term.first;
        } else {
            this->add_term(-term.first, term.second);
        }
    }
    return *this;
}

Observable Observable::operator*(const Observable& target) const {
    Observable res;
    ITYPE i, j;
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
    Observable res = *(this->copy());
    res *= target;
    return res;
}

Observable& Observable::operator*=(const Observable& target) {
    Observable tmp = (*this) * target;
    this->_coef_list.clear();
    this->_pauli_terms.clear();
    ITYPE i;
    for (i = 0; i < tmp.get_term_count(); i++) {
        auto term = tmp.get_term(i);
        this->add_term(term.first, term.second);
    }
    return *this;
}

Observable& Observable::operator*=(const CPPCTYPE& target) {
    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < this->_coef_list.size(); i++) {
        this->_coef_list[i] *= target;
    }
    return *this;
}

std::string Observable::to_string() const{
    std::ostringstream ss;
    std::string res;
    ITYPE i;
    for (i = 0; i < get_term_count(); i++) {
        // (1.0-2.0j)
        ss << "(" << _coef_list[i].real();
        // 虚数部には符号をつける
        // +0j or -0j に対応させるためstd::showposを用いる
        ss << std::showpos << _coef_list[i].imag() << "j) ";

        // [X 0 Y 1 Z 2]
        ss << "[" << _pauli_terms[i].to_string() << "]";
        if (i != get_term_count() - 1) {
            ss << " +" << std::endl;
        }
        res += ss.str();
        ss.str("");
        ss.clear();
        // noshowposして、1項目以降の実数部に符号が付かないようにする
        ss << std::noshowpos;
    }
    return res;
}