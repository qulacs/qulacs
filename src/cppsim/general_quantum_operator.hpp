#pragma once

#include <cstdio>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "type.hpp"
class PauliOperator;
class QuantumStateBase;

class DllExport GeneralQuantumOperator {
private:
    //! list of multi pauli term
    std::vector<PauliOperator*> _operator_list;
    //! the number of qubits
    UINT _qubit_count;
    bool _is_hermitian;

public:
    /**
     * \~japanese-en
     * コンストラクタ。
     *
     * 空のGeneralQuantumOperatorを作成する。
     * @param[in] qubit_count qubit数
     * @return Observableのインスタンス
     */
    explicit GeneralQuantumOperator(const UINT qubit_count);

    /**
     * \~japanese-en
     * デストラクタ。このとき、GeneralQuantumOperatorが保持しているPauliOperatorは解放される。
     */
    virtual ~GeneralQuantumOperator();

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    virtual bool is_hermitian() const { return _is_hermitian; }

    /**
     * \~japanese-en
     * PauliOperatorを内部で保持するリストの末尾に追加する。
     *
     * @param[in] mpt 追加するPauliOperatorのインスタンス
     */
    virtual void add_operator(const PauliOperator* mpt);

    /**
     * \~japanese-en
     * パウリ演算子の文字列と係数の組をGeneralQuantumOperatorに追加する。
     *
     * @param[in] coef pauli_stringで作られるPauliOperatorの係数
     * @param[in] pauli_string
     * パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    virtual void add_operator(const CPPCTYPE coef, std::string pauli_string);

    /**
     * \~japanese-en
     * GeneralQuantumOperatorが掛かるqubit数を返す。
     * @return GeneralQuantumOperatorのqubit数
     */
    virtual UINT get_qubit_count() const { return _qubit_count; }

    /**
     * \~japanese-en
     * GeneralQuantumOperatorの行列表現の次元を返す。
     * @return GeneralQuantumOperatorの次元
     */
    virtual ITYPE get_state_dim() const { return (1ULL) << _qubit_count; }

    /**
     * \~japanese-en
     * GeneralQuantumOperatorが保持するPauliOperatorの数を返す
     * @return GeneralQuantumOperatorが保持するPauliOperatorの数
     */
    virtual UINT get_term_count() const { return (UINT)_operator_list.size(); }

    /**
     * \~japanese-en
     * GeneralQuantumOperatorの指定した添字に対応するPauliOperatorを返す
     * @param[in] index
     * GeneralQuantumOperatorが保持するPauliOperatorのリストの添字
     * @return 指定したindexにあるPauliOperator
     */
    virtual const PauliOperator* get_term(UINT index) const {
        if (index >= _operator_list.size()) {
            std::stringstream error_message_stream;
            error_message_stream
                << "Error: PauliOperator::get_term(UINT): index out of range";
            throw std::out_of_range(error_message_stream.str());
        }
        return _operator_list[index];
    }

    /**
     * \~japanese-en
     * GeneralQuantumOperatorが保持するPauliOperatorのリストを返す
     * @return GeneralQuantumOperatorが持つPauliOperatorのリスト
     */
    virtual std::vector<PauliOperator*> get_terms() const {
        return _operator_list;
    }

    /**
     * \~japanese-en
     * 文字列に変換する。
     */
    virtual std::string to_string() const;

    /**
     * \~japanese-en
     * GeneralQuantumOperatorのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するGeneralQuantumOperatorの期待値
     */
    virtual CPPCTYPE get_expectation_value(const QuantumStateBase* state) const;

    /**
     * \~japanese-en
     * GeneralQuantumOperatorによってある状態が別の状態に移る遷移振幅を計算して返す
     *
     * @param[in] state_bra 遷移先の量子状態
     * @param[in] state_ket 遷移前の量子状態
     * @return 入力で与えた量子状態に対応するGeneralQuantumOperatorの遷移振幅
     */
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra,
        const QuantumStateBase* state_ket) const;

    /**
     * \~japanese-en
     * ランダムなパウリ演算子をもつ observable を生成する
     * @param [in] observable パウリ演算子を追加する observable
     * @param [in] operator_count observable に追加するパウリ演算子数
     * @return ランダムなパウリ演算子を operator_count 個もつ　observable
     */
    void add_random_operator(const UINT operator_count);

    /**
     * \~japanese-en
     * GeneralQuantumOperator の基底状態の固有値を arnordi method により求める．
     * (A - \mu I) の絶対値最大固有値を求めることで基底状態の固有値を求める．
     * @param[in] state 固有値を求めるための量子状態
     * @param[in] iter_count 計算の繰り返し回数
     * @param [in] mu 固有値をシフトするための係数
     * @return GeneralQuantumOperator の基底状態の固有値
     */
    virtual CPPCTYPE solve_ground_state_eigenvalue_by_arnoldi_method(
        QuantumStateBase* state, const UINT iter_count,
        const CPPCTYPE mu = 0.0) const;

    /**
     * \~japanese-en
     * GeneralQuantumOperator の基底状態の固有値を power method により求める
     * (A - \mu I) の絶対値最大固有値を求めることで基底状態の固有値を求める．
     * @param[in] state 固有値を求めるための量子状態
     * @param[in] iter_count 計算の繰り返し回数
     * @param [in] mu 固有値をシフトするための係数
     * @return GeneralQuantumOperator の基底状態の固有値
     */
    virtual CPPCTYPE solve_ground_state_eigenvalue_by_power_method(
        QuantumStateBase* state, const UINT iter_count,
        const CPPCTYPE mu = 0.0) const;

    /**
     * \~japanese-en
     * state_to_be_multiplied に GeneralQuantumOperator を作用させる．
     * 結果は dst_state に格納される．dst_state
     * はすべての要素を0に初期化してから計算するため， 任意の状態を渡してよい．
     * @param [in] work_state 作業用の状態
     * @param [in] state_to_be_multiplied 作用を受ける状態
     * @param [in] dst_state 結果を格納する状態
     */
    void apply_to_state(QuantumStateBase* work_state,
        const QuantumStateBase& state_to_be_multiplied,
        QuantumStateBase* dst_state) const;

    virtual GeneralQuantumOperator* copy() const;

    GeneralQuantumOperator operator+(
        const GeneralQuantumOperator& target) const;

    GeneralQuantumOperator operator+(const PauliOperator& target) const;

    GeneralQuantumOperator& operator+=(const GeneralQuantumOperator& target);

    GeneralQuantumOperator& operator+=(const PauliOperator& target);

    GeneralQuantumOperator operator-(
        const GeneralQuantumOperator& target) const;

    GeneralQuantumOperator operator-(const PauliOperator& target) const;

    GeneralQuantumOperator& operator-=(const GeneralQuantumOperator& target);

    GeneralQuantumOperator& operator-=(const PauliOperator& target);
    GeneralQuantumOperator operator*(
        const GeneralQuantumOperator& target) const;

    GeneralQuantumOperator operator*(const PauliOperator& target) const;

    GeneralQuantumOperator operator*(CPPCTYPE target) const;

    GeneralQuantumOperator& operator*=(const GeneralQuantumOperator& target);

    GeneralQuantumOperator& operator*=(const PauliOperator& target);

    GeneralQuantumOperator& operator*=(CPPCTYPE target);

protected:
    /**
     * \~japanese-en
     * solve_ground_state_eigenvalue_by_power_method の mu
     * のデフォルト値として，各 operator の係数の絶対値の和を計算する．
     */
    CPPCTYPE calculate_default_mu() const;
};

namespace quantum_operator {
/**
 * \~japanese-en
 *
 * OpenFermionから出力されたGeneralQuantumOperatorのテキストファイルを読み込んでGeneralQuantumOperatorを生成します。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
 *
 * @param[in] filename OpenFermion形式のGeneralQuantumOperatorのファイル名
 * @return Observableのインスタンス
 **/
DllExport GeneralQuantumOperator*
create_general_quantum_operator_from_openfermion_file(std::string file_path);

/**
 * \~japanese-en
 *
 * OpenFermionの出力テキストを読み込んでGeneralQuantumOperatorを生成します。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
 *
 * @param[in] filename OpenFermion形式のテキスト
 * @return General_Quantum_Operatorのインスタンス
 **/
DllExport GeneralQuantumOperator*
create_general_quantum_operator_from_openfermion_text(std::string text);

/**
 * \~japanese-en
 * OpenFermion形式のファイルを読んで、対角なGeneralQuantumOperatorと非対角なGeneralQuantumOperatorを返す。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
 *
 * @param[in] filename OpenFermion形式のGeneralQuantumOperatorのファイル名
 */
DllExport std::pair<GeneralQuantumOperator*, GeneralQuantumOperator*>
create_split_general_quantum_operator(std::string file_path);
}  // namespace quantum_operator

bool check_Pauli_operator(const GeneralQuantumOperator* quantum_operator,
    const PauliOperator* pauli_operator);
