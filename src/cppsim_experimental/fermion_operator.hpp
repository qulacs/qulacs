#pragma once

#include "single_fermion_operator.hpp"
#include "state.hpp"
#include "type.hpp"

class DllExport FermionOperator {
private:
    std::vector<SingleFermionOperator> _fermi_terms;
    std::vector<CPPCTYPE> _coef_list;

public:
    FermionOperator();

    UINT get_term_count() const;

    std::pair<CPPCTYPE, SingleFermionOperator> get_term(const UINT index) const;
    /**
     * SingleFermionOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] coef 係数
     * @param[in] fermion_operator 追加するSingleFermionOperatorのインスタンス
     */
    void add_term(const CPPCTYPE coef, SingleFermionOperator fermion_operator);

    /**
     * フェルミオン演算子の文字列と係数の組を追加する。
     *
     * @param[in] coef
     * @param[in] action_string
     * 演算子と掛かるindexの組からなる文字列。(example: "1 2^ 3 5^")
     */
    void add_term(const CPPCTYPE coef, std::string action_string);

    void remove_term(UINT index);
};
