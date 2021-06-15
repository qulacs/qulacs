#include "fermion_operator.hpp"

#include <boost/range/combine.hpp>

#include "observable.hpp"
#include "pauli_operator.hpp"
#include "state.hpp"
#include "type.hpp"

FermionOperator::FermionOperator(){};

UINT FermionOperator::get_term_count() const {
    return (UINT)_fermion_terms.size();
}

std::pair<CPPCTYPE, SingleFermionOperator> FermionOperator::get_term(
    const UINT index) const {
    return std::make_pair(
        this->_coef_list.at(index), this->_fermion_terms.at(index));
}

void FermionOperator::add_term(
    const CPPCTYPE coef, SingleFermionOperator fermion_operator) {
    this->_coef_list.push_back(coef);
    this->_fermion_terms.push_back(fermion_operator);
}

void FermionOperator::add_term(const CPPCTYPE coef, std::string action_string) {
    this->_coef_list.push_back(coef);
    this->_fermion_terms.push_back(SingleFermionOperator(action_string));
}

void FermionOperator::remove_term(UINT index) {
    this->_coef_list.erase(this->_coef_list.begin() + index);
    this->_fermion_terms.erase(this->_fermion_terms.begin() + index);
}

const std::vector<SingleFermionOperator>& FermionOperator::get_fermion_list()
    const {
    return _fermion_terms;
}

const std::vector<CPPCTYPE>& FermionOperator::get_coef_list() const {
    return _coef_list;
}

Observable FermionOperator::jordan_wigner() {
    Observable observable;
    for (auto fop_tuple : boost::combine(_fermion_terms, _coef_list)) {
        SingleFermionOperator sfop;
        CPPCTYPE coef;
        boost::tie(sfop, coef) = fop_tuple;

        MultiQubitPauliOperator qubit_operator;

        auto target_index_list = sfop.get_target_index_list();
        auto action_id_list = sfop.get_action_id_list();
        for (auto ladder_operator :
            boost::combine(target_index_list, action_id_list)) {
            UINT target_index;
            UINT action_id;
            boost::tie(target_index, action_id) = ladder_operator;

            if (target_index == 0) {
                observable.add_term(coef * 0.5, MultiQubitPauliOperator("X 0"));
                CPPCTYPE coef_Y = coef * CPPCTYPE(0, 0.5);
                if (action_id) coef_Y *= -1;
                observable.add_term(coef_Y, MultiQubitPauliOperator("Y 0"));
                continue;
            }
            // Z factors
            std::vector<UINT> target_qubit_index_list(target_index);
            std::vector<UINT> pauli_id_list(target_index, PAULI_ID_Z);
            for (UINT i = 0; i < target_index; i++) {
                target_qubit_index_list[i] = i + 1;
            }

            // X factors
            pauli_id_list.at(target_index - 1) = PAULI_ID_X;
            observable.add_term(
                coef * 0.5, MultiQubitPauliOperator(
                                target_qubit_index_list, pauli_id_list));

            // Y factors
            pauli_id_list.at(target_index - 1) = PAULI_ID_Y;
            if (action_id)
                observable.add_term(-coef * CPPCTYPE(0, 0.5),
                    MultiQubitPauliOperator(
                        target_qubit_index_list, pauli_id_list));
            else
                observable.add_term(coef * CPPCTYPE(0, 0.5),
                    MultiQubitPauliOperator(
                        target_qubit_index_list, pauli_id_list));
        }
    }

    return observable;
}