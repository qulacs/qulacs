#pragma once

#include "observable.hpp"
#include "single_fermion_operator.hpp"
#include "state.hpp"
#include "type.hpp"

/**
 * \~japanese-en
 * @struct FermionOperator
 * オブザーバブルのように、SingleFermionOperatorをリストとして持ち、操作を行うクラス。
 */
class DllExport FermionOperator {
private:
    std::vector<SingleFermionOperator> _fermion_terms;
    std::vector<CPPCTYPE> _coef_list;

public:
    FermionOperator();

    /**
     * FermionOperator が保持する SingleFermionOperator の個数を返す
     * @return FermionOperator が保持する SingleFermionOperator の個数
     */
    UINT get_term_count() const;

    /**
     * FermionOperator の指定した添字に対応するSingleFermionOperatorを返す
     * @param[in] index
     * FermionOperator が保持するSingleFermionOperatorのリストの添字
     * @return 指定したindexにあるSingleFermionOperator
     */
    std::pair<CPPCTYPE, SingleFermionOperator> get_term(const UINT index) const;
    /**
     * SingleFermionOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] coef 係数
     * @param[in] fermion_operator 追加するSingleFermionOperatorのインスタンス
     */
    void add_term(const CPPCTYPE coef, SingleFermionOperator fermion_operator);

    /**
     * 係数とフェルミオン演算子の文字列の組を追加する。
     *
     * @param[in] coef
     * @param[in] action_string
     * 演算子と作用する軌道の添字の組からなる文字列。(example: "2^ 1")
     */
    void add_term(const CPPCTYPE coef, std::string action_string);

    /**
     * 指定した添字に対応するフェルミオン演算子の項を削除する
     * @param[in] index
     * FermionOperator が保持するSingleFermionOperatorのリストの添字
     */
    void remove_term(UINT index);

    /**
     * 各項のSingleFermionOperatorのリストを取得する
     */
    const std::vector<SingleFermionOperator>& get_fermion_list() const;

    /**
     * 各項の係数のリストを取得する
     */
    const std::vector<CPPCTYPE>& get_coef_list() const;

    /**
     * Jordan-Wigner変換を適用してFermionOperatorをQubitOperatorに変換する
     */
    Observable jordan_wigner();
    Observable bravyi_kitaev(UINT n_qubits);
};

UINT count_qubits(FermionOperator& fop);
Observable inline_sum(std::vector<Observable> summands, Observable seed);
Observable inline_product(std::vector<Observable> factors, Observable seed);
std::set<UINT> _update_set(UINT index, UINT n_qubits);
std::set<UINT> _occupation_set(UINT index);
std::set<UINT> _parity_set(UINT index);
Observable _transform_ladder_operator(
    UINT target_index, UINT action_id, UINT n_qubits);
Observable _transform_operator_term(
    SingleFermionOperator& sfop, CPPCTYPE coef, UINT n_qubits);
