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
    this->_term_dict[fermion_operator.to_string()] = _coef_list.size() - 1;
}

void FermionOperator::add_term(const CPPCTYPE coef, std::string action_string) {
    this->_coef_list.push_back(coef);
    this->_fermion_terms.push_back(SingleFermionOperator(action_string));
    this->_term_dict[action_string] = _coef_list.size() - 1;
}

void FermionOperator::add_term(const std::vector<CPPCTYPE> coef_list,
    std::vector<SingleFermionOperator> fermion_terms) {
    int base_size = this->_coef_list.size();
    int changed_size = base_size + coef_list.size();
    // vectorをindexでアクセス出来るようにする
    this->_coef_list.resize(changed_size);
    this->_fermion_terms.resize(changed_size);

#pragma omp parallel
    {
        std::unordered_map<std::string, ITYPE> term_dict_private;
#pragma omp for nowait
        for (ITYPE index = 0; index < coef_list.size(); index++) {
            int insert_pos = base_size + index;
            this->_coef_list[insert_pos] = coef_list[index];
            this->_fermion_terms[insert_pos] = fermion_terms[index];
            // dictには並列で同時に書き込めないので、一旦term_dict_privateに書き込む
            term_dict_private[fermion_terms[index].to_string()] = insert_pos;
        }

#pragma omp critical
        {
            _term_dict.insert(
                term_dict_private.begin(), term_dict_private.end());
        }
    }
}

void FermionOperator::remove_term(UINT index) {
    this->_term_dict.erase(this->_fermion_terms.at(index).to_string());
    this->_coef_list.erase(this->_coef_list.begin() + index);
    this->_fermion_terms.erase(this->_fermion_terms.begin() + index);

    // index番目の項を削除したので、index番目以降の項のindexが1つずれる
    for (ITYPE i = 0; i < this->_coef_list.size() - index; i++) {
        this->_term_dict[this->_fermion_terms.at(index + i).to_string()] =
            index + i;
    }
}

const std::vector<SingleFermionOperator>& FermionOperator::get_fermion_list()
    const {
    return _fermion_terms;
}

const std::vector<CPPCTYPE>& FermionOperator::get_coef_list() const {
    return _coef_list;
}

const std::unordered_map<std::string, ITYPE>& FermionOperator::get_dict()
    const {
    return _term_dict;
}

FermionOperator* FermionOperator::copy() const {
    FermionOperator* res = new FermionOperator();
    res->add_term(this->_coef_list, this->_fermion_terms);
    return res;
}

FermionOperator FermionOperator::operator+(
    const FermionOperator& target) const {
    FermionOperator res = *(this->copy());
    res += target;
    return res;
}

FermionOperator FermionOperator::operator+=(const FermionOperator& target) {
    auto u_map = target.get_dict();

    for (auto item : u_map) {
        if (_term_dict.find(item.first) != _term_dict.end()) {
            ITYPE id = _term_dict[item.first];
            this->_coef_list[id] += target.get_term(item.second).first;
        } else {
            auto term = target.get_term(item.second);
            this->add_term(term.first, term.second);
        }
    }
    return *this;
}

FermionOperator FermionOperator::operator-(
    const FermionOperator& target) const {
    FermionOperator res = *(this->copy());
    res -= target;
    return res;
}

FermionOperator FermionOperator::operator-=(const FermionOperator& target) {
    auto u_map = target.get_dict();

    for (auto item : u_map) {
        if (_term_dict.find(item.first) != _term_dict.end()) {
            ITYPE id = _term_dict[item.first];
            this->_coef_list[id] -= target.get_term(item.second).first;
        } else {
            auto term = target.get_term(item.second);
            this->add_term(-term.first, term.second);
        }
    }
    return *this;
}

FermionOperator FermionOperator::operator*(
    const FermionOperator& target) const {
    FermionOperator res;
    ITYPE i, j;
    for (i = 0; i < this->_fermion_terms.size(); i++) {
        for (j = 0; j < target.get_term_count(); j++) {
            FermionOperator tmp;
            auto term = target.get_term(j);
            tmp.add_term(this->_coef_list[i] * term.first,
                this->_fermion_terms[i] * term.second);
            res += tmp;
        }
    }
    return res;
}

FermionOperator FermionOperator::operator*(const CPPCTYPE& target) const {
    FermionOperator res = *(this->copy());
    res *= target;
    return res;
}

FermionOperator FermionOperator::operator*=(const FermionOperator& target) {
    FermionOperator tmp = (*this) * target;
    this->_coef_list.clear();
    this->_fermion_terms.clear();
    this->_term_dict.clear();
    ITYPE i;
    this->add_term(tmp.get_coef_list(), tmp.get_fermion_list());
    return *this;
}

FermionOperator FermionOperator::operator*=(const CPPCTYPE& target) {
    ITYPE i;
#pragma omp parallel for
    for (i = 0; i < this->_coef_list.size(); i++) {
        this->_coef_list[i] *= target;
    }
    return *this;
}
