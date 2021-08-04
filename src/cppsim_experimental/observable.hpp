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
    std::unordered_map<std::string, ITYPE> _term_dict;

    CPPCTYPE calc_coef(const MultiQubitPauliOperator& a,
        const MultiQubitPauliOperator& b) const;

public:
    Observable(){};

    /**
     * Observable が保持する PauliOperator の個数を返す
     * @return Observable が保持する PauliOperator の個数
     */
    UINT get_term_count() const { return (UINT)_pauli_terms.size(); }

    /**
     * Observable の指定した添字に対応するPauliOperatorを返す
     * @param[in] index
     * Observable が保持するPauliOperatorのリストの添字
     * @return 指定したindexにあるPauliOperator
     */
    std::pair<CPPCTYPE, MultiQubitPauliOperator> get_term(
        const UINT index) const;

    /**
     * Observable が保持するPauliOperatorをunordered_mapとして返す
     * @return Observable が保持するunordered_map
     */
    std::unordered_map<std::string, ITYPE> get_dict() const;

    /**
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    void add_term(const CPPCTYPE coef, MultiQubitPauliOperator op);

    /**
     * パウリ演算子の文字列と係数の組をObservable に追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string
     * パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    void add_term(const CPPCTYPE coef, std::string s);

    void remove_term(UINT index);

    /**
     * Observable のある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するObservable の期待値
     */
    CPPCTYPE get_expectation_value(const QuantumStateBase* state) const;

    /**
     * Observable によってある状態が別の状態に移る遷移振幅を計算して返す
     *
     * @param[in] state_bra 遷移先の量子状態
     * @param[in] state_ket 遷移前の量子状態
     * @return 入力で与えた量子状態に対応するObservable の遷移振幅
     */
    CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const;

    /**
     * ヒープに確保した Observable を返す
     *
     * @return ヒープに確保した Observable へのポインタ
     */
    Observable* copy() const;

    Observable operator+(const Observable& target) const;

    Observable& operator+=(const Observable& target);

    Observable operator-(const Observable& target) const;

    Observable& operator-=(const Observable& target);

    Observable operator*(const Observable& target) const;

    Observable operator*(const CPPCTYPE& target) const;

    Observable& operator*=(const Observable& target);

    Observable& operator*=(const CPPCTYPE& target);

    /**
     * Observableを文字列に変換する
     */
    std::string to_string() const;
};
