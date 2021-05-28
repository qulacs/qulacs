#include "fermion_operator.hpp"

#include "state.hpp"
#include "type.hpp"

UINT FermionOperator::get_term_count() const {
    return (UINT)_fermi_terms.size();
}

std::pair<CPPCTYPE, SingleFermionOperator> FermionOperator::get_term(
    const UINT index) const {
    return std::make_pair(
        this->_coef_list.at(index), this->_fermi_terms.at(index));
}

void FermionOperator::add_term(const CPPCTYPE coef, SingleFermionOperator op) {
    this->_coef_list.push_back(coef);
    this->_fermi_terms.push_back(op);
}

void FermionOperator::add_term(const CPPCTYPE coef, std::string action_string) {
    this->_coef_list.push_back(coef);
    this->_fermi_terms.push_back(SingleFermionOperator(action_string));
}

void FermionOperator::remove_term(UINT index) {
    this->_coef_list.erase(this->_coef_list.begin() + index);
    this->_fermi_terms.erase(this->_fermi_terms.begin() + index);
}