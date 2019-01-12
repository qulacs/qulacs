#pragma once

#include "type.hpp"
#include <cstdio>
#include <vector>
#include <utility>
#include <iostream>
#include <string>

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
    GeneralQuantumOperator(UINT qubit_count);

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
     * @param[in] pauli_string パウリ演算子と掛かるindexの組からなる文字列。(example: "X 1 Y 2 Z 5")
     */
    virtual void add_operator(CPPCTYPE coef, std::string pauli_string);

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
     * @param[in] index GeneralQuantumOperatorが保持するPauliOperatorのリストの添字
     * @return 指定したindexにあるPauliOperator
     */
    virtual const PauliOperator* get_term(UINT index) const {
		if (index >= _operator_list.size()) {
			std::cerr << "Error: PauliOperator::get_term(UINT): index out of range" << std::endl;
			return NULL;
		}
		return _operator_list[index];
	}

    /**
     * \~japanese-en
     * GeneralQuantumOperatorが保持するPauliOperatorのリストを返す
     * @return GeneralQuantumOperatorが持つPauliOperatorのリスト
     */
    virtual std::vector<PauliOperator*> get_terms() const { return _operator_list;}

    /**
     * \~japanese-en
     * GeneralQuantumOperatorのある量子状態に対応するエネルギー(期待値)を計算して返す
     *
     * @param[in] state 期待値をとるときの量子状態
     * @return 入力で与えた量子状態に対応するGeneralQuantumOperatorの期待値
     */
    virtual CPPCTYPE get_expectation_value(const QuantumStateBase* state) const ;

    /**
     * \~japanese-en
     * GeneralQuantumOperatorによってある状態が別の状態に移る遷移振幅を計算して返す
     *
     * @param[in] state_bra 遷移先の量子状態
     * @param[in] state_ket 遷移前の量子状態
     * @return 入力で与えた量子状態に対応するGeneralQuantumOperatorの遷移振幅
     */
    virtual CPPCTYPE get_transition_amplitude(const QuantumStateBase* state_bra, const QuantumStateBase* state_ket) const;

};

namespace quantum_operator{
    /**
     * \~japanese-en
     *
     * OpenFermionから出力されたGeneralQuantumOperatorのテキストファイルを読み込んでGeneralQuantumOperatorを生成します。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のGeneralQuantumOperatorのファイル名
     * @return Observableのインスタンス
     **/
    DllExport GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_file(std::string file_path);

    /**
     * \~japanese-en
     *
     * OpenFermionの出力テキストを読み込んでGeneralQuantumOperatorを生成します。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のテキスト
     * @return General_Quantum_Operatorのインスタンス
     **/
    DllExport GeneralQuantumOperator* create_general_quantum_operator_from_openfermion_text(std::string text);

    /**
     * \~japanese-en
     * OpenFermion形式のファイルを読んで、対角なGeneralQuantumOperatorと非対角なGeneralQuantumOperatorを返す。GeneralQuantumOperatorのqubit数はファイル読み込み時に、GeneralQuantumOperatorの構成に必要なqubit数となります。
     *
     * @param[in] filename OpenFermion形式のGeneralQuantumOperatorのファイル名
     */
    DllExport std::pair<GeneralQuantumOperator*, GeneralQuantumOperator*> create_split_general_quantum_operator(std::string file_path);
}

bool check_Pauli_operator(const GeneralQuantumOperator* quantum_operator, const PauliOperator* pauli_operator);
