
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
        const MultiQubitPauliOperator& b) const;

public:
    Observable(){};

    UINT get_term_count() const { return (UINT)_pauli_terms.size(); }

    std::pair<CPPCTYPE, MultiQubitPauliOperator> get_term(
        const UINT index) const;

    void add_term(const CPPCTYPE coef, MultiQubitPauliOperator op);

    void add_term(const CPPCTYPE coef, std::string s);

    void remove_term(UINT index);

    CPPCTYPE get_expectation_value(
        const QuantumStateBase* state) const;

    CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const;

    Observable* copy() const;

    Observable operator+(const Observable& target) const;

    Observable& operator+=(const Observable& target);

    Observable operator-(const Observable& target) const;

    Observable& operator-=(const Observable& target);

    Observable operator*(const Observable& target) const;

    Observable operator*(const CPPCTYPE& target) const;

    Observable& operator*=(const Observable& target);

    Observable& operator*=(const CPPCTYPE& target);
};
